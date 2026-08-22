"""Older Amin shared-memory driver. One process, whole cycle.

The production steps live in src/agent_memories/generalisation. This
file still runs N private trajectories, then Amin labels, bucketing,
content, and title/description, and writes data/memories/shared.jsonl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent_memories.agent.amin_et_al import (
    PrivacyAccount,
    epsilon_from_rho,
    generate,
    rho_for,
    solve_r,
)
from agent_memories.agent.lm.prompts import wrap, wrap_label
from agent_memories.config import DEFAULT_MEMORY_DIR, load_random_seed
from agent_memories.generalisation import (
    assign_memories_to_labels,
    entry_id,
    group_by_label,
    parse_json_labels,
    read_buffer,
    select_round2_inputs,
    title_and_description,
    write_buffer,
)
from agent_memories.memory import Embedder, MemoryEntry, MemoryItem, MemoryStore
from agent_memories.services.ollama_client import OllamaClient

REPO_ROOT = Path(__file__).resolve().parents[2]
WP1_5_SCRIPT = REPO_ROOT / "scripts" / "memories" / "WP1_5.py"

N_TRAJECTORIES = 5
K_LABELS = 3
X_PER_LABEL = 5

AIM_DEFAULT = "Find cheapest hairbrush on Amazon"
URL_DEFAULT = "https://www.amazon.co.uk"
USER_PREFIX = "user_"
MAX_STEPS_DEFAULT = 10
MODEL_DEFAULT = "qwen"

QWEN_MODEL = "qwen3.5:4b-nvfp4"

# Amin Algorithm 1 knobs; both rounds reuse this set.
C = 50.0
TAU = 1.0
TAU_PUBLIC = 1.5
SIGMA = 0.1
THETA = 0.0
R_MAX = 80
EPSILON_PER_ROUND = 200.0
DELTA = 1e-5

SHARED_STORE_FILENAME = "shared.jsonl"
BUFFER_FILENAME = ".shared_buffer.jsonl"
CHECKPOINT_FILENAME = ".shared_state.json"
AUDIT_FILENAME = ".shared_audit.jsonl"
CHECKPOINT_SCHEMA_VERSION = 3


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _render_item_block(entry: MemoryEntry) -> str:
    """Render one trajectory entry's items for a round-2 prompt row."""
    return "\n".join(
        f"Memory {index}: {item.title} | {item.description} | {item.content}"
        for index, item in enumerate(entry.items, start=1)
    )


def _render_query_block(entry: MemoryEntry) -> str:
    """Parent query as the round-1 prompt row.

    Round 1 produces the DP labels that become the ``query`` (and
    therefore the embedding) of every shared memory. Feeding the
    trajectory aim, not the item block, keeps those labels in the
    same space as future task aims. Round 2 still uses
    ``_render_item_block`` because its output is ``content``, which
    needs the actual reasoning steps.
    """
    return entry.query


def _run_trajectories(
    *,
    n: int,
    aim: str,
    url: str,
    user_prefix: str,
    memory_dir: Path,
    max_steps: int,
    headless: bool,
    model: str,
) -> list[str]:
    """Spawn ``n`` WP1.5 subprocesses; return the per-trajectory user_ids.

    Each trajectory writes ``data/memories/<user_id>.jsonl``. A
    non-zero exit is logged and the loop continues; later steps
    consume whichever stores actually have entries.
    """
    user_ids = [f"{user_prefix}{i:02d}" for i in range(n)]
    for idx, user_id in enumerate(user_ids):
        print("=" * 72)
        print(f"[WP2] Trajectory {idx + 1} / {n} (user_id={user_id})")
        print("=" * 72)
        cmd = [
            sys.executable,
            str(WP1_5_SCRIPT),
            "--aim",
            aim,
            "--url",
            url,
            "--user-id",
            user_id,
            "--max-steps",
            str(max_steps),
            "--model",
            model,
            "--memory-dir",
            str(memory_dir),
        ]
        if headless:
            cmd.append("--headless")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[WP2] WP1_5.py exited {result.returncode} for {user_id}; continuing.")
    return user_ids


def _load_checkpoint(path: Path) -> dict:
    """Return the persisted checkpoint, or a fresh one if the file is missing."""
    if not path.exists():
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "last_run_at": None,
            "step1_processed_entry_ids": [],
            "consumed_entry_ids": [],
            "trigger_count": 0,
            "cumulative_epsilon": 0.0,
            "cumulative_delta": 0.0,
        }
    state: dict = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            "Checkpoint uses the legacy item-level schema. Reset "
            ".shared_state.json and .shared_buffer.jsonl before running "
            "entry-level privacy."
        )
    return state


def _save_checkpoint(path: Path, state: dict) -> None:
    """Persist the checkpoint as pretty-printed JSON for hand-readability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _append_audit(path: Path, record: dict) -> None:
    """Append one JSONL audit line; the file is the privacy-spend ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_per_user_entries(
    memory_dir: Path,
    step1_processed_ids: set[str],
    embedder: Embedder,
) -> list[MemoryEntry]:
    """Return trajectory entries that have not yet completed round 1.

    Only ``user_*.jsonl``; ``shared.jsonl`` and hidden sidecars
    (``.shared_buffer.jsonl`` and friends) are skipped so the input
    set is private per-user memories only.
    """
    new_entries: list[MemoryEntry] = []
    for path in sorted(memory_dir.glob("user_*.jsonl")):
        store = MemoryStore.load(path, user_id=path.stem, embedder=embedder)
        for entry in store.all():
            if entry.embedding is None:
                raise ValueError(
                    f"Memory entry for user_id={entry.user_id!r}, query={entry.query!r} "
                    "has no query embedding."
                )
            if entry_id(entry) not in step1_processed_ids:
                new_entries.append(entry)
    return new_entries


def _assemble_cycle_entries(
    memory_dir: Path,
    step1_processed_ids: set[str],
    carry_over: list[MemoryEntry],
    embedder: Embedder,
    *,
    entries_per_cycle: int,
) -> tuple[list[MemoryEntry], list[MemoryEntry], set[str]]:
    """Return one complete round-1 entry batch and round-2 candidates.

    Carry-over entries prove that they completed round 1 in an earlier
    trigger. Adding their IDs before loading the user stores prevents the
    same protected examples from appearing once as new and once as carry-over.
    Fewer than ``entries_per_cycle`` new examples remain pending in their
    source stores; no partial round-1 cycle is released.
    """
    processed_ids = set(step1_processed_ids)
    processed_ids.update(entry_id(entry) for entry in carry_over)
    available = _load_per_user_entries(memory_dir, processed_ids, embedder)
    if len(available) < entries_per_cycle:
        return [], [], processed_ids
    new_entries = available[:entries_per_cycle]
    return new_entries, new_entries + carry_over, processed_ids


def _print_round_summary(
    label: str,
    *,
    s: int,
    delta: float,
    r: int,
    eps: float,
    target_epsilon: float,
) -> None:
    print(
        f"[WP2] {label}: S={s}, delta={delta:.4f}, r={r}, "
        f"eps={eps:.4f} (target={target_epsilon})"
    )


def _run_round1(
    round1_batch: list[MemoryEntry],
    *,
    expected_batch_size: int,
    k_labels: int,
    target_epsilon: float,
    delta: float,
) -> tuple[list[str], list[bool], int, float, float]:
    """Run round 1 once; return labels, parsed flags, r, eps, delta.

    Every entry is one prompt row. Accounting uses the public fixed
    cycle size, not the realised private counts.
    """
    s = expected_batch_size
    if len(round1_batch) != s:
        raise ValueError(f"Round 1 requires exactly B={s} entries; got {len(round1_batch)}.")
    r = solve_r(target_epsilon, delta, s=s, c=C, tau=TAU, sigma=SIGMA, r_max=R_MAX)
    if r == 0:
        raise RuntimeError(
            f"Round 1 budget too tight: target_epsilon={target_epsilon}, "
            f"delta={delta}, S={s} -> r=0. Loosen EPSILON_PER_ROUND or reduce sigma."
        )
    eps = epsilon_from_rho(rho_for(r, s, C, TAU, SIGMA), delta)
    _print_round_summary("Round 1", s=s, delta=delta, r=r, eps=eps, target_epsilon=target_epsilon)

    items_blocks = [_render_query_block(entry) for entry in round1_batch]
    raw, _account = generate(
        items_blocks,
        s=s,
        c=C,
        tau=TAU,
        tau_public=TAU_PUBLIC,
        sigma=SIGMA,
        theta=THETA,
        r=r,
        delta=delta,
        wrap_fn=wrap_label,
        k=k_labels,
    )
    parsed = parse_json_labels(raw, k_labels)
    labels = [text for text, _ in parsed]
    flags = [ok for _, ok in parsed]

    print(f"[WP2] Round 1 raw DP output: {raw!r}")
    print(f"[WP2] Round 1 parsed labels (k={k_labels}):")
    for idx, (text, ok) in enumerate(parsed, start=1):
        tag = "" if ok else "  [fallback]"
        print(f"    {idx}. {text}{tag}")
    return labels, flags, r, eps, delta


def _run_round2_for_label(
    *,
    label: str,
    entries: list[MemoryEntry],
    expected_batch_size: int,
    target_epsilon: float,
    delta: float,
) -> tuple[str, int, float, float, PrivacyAccount] | None:
    """One round-2 generate call; DP content plus the privacy account.

    Every entry under this label is included. Accounting still uses
    the fixed X threshold as expected batch size; that need not match
    the actual batch.

    Returns None if solve_r cannot fit even one private token, so
    the caller can skip post-processing for this label without
    aborting the trigger.
    """
    s = expected_batch_size
    r = solve_r(target_epsilon, delta, s=s, c=C, tau=TAU, sigma=SIGMA, r_max=R_MAX)
    if r == 0:
        print(
            f"[WP2] Round 2 label {label!r}: budget too tight "
            f"(target={target_epsilon}, delta={delta}, S={s} -> r=0). Skipping."
        )
        return None
    eps = epsilon_from_rho(rho_for(r, s, C, TAU, SIGMA), delta)
    _print_round_summary(
        f"Round 2 [{label!r}]",
        s=s,
        delta=delta,
        r=r,
        eps=eps,
        target_epsilon=target_epsilon,
    )

    items_blocks = [_render_item_block(entry) for entry in entries]
    content, account = generate(
        items_blocks,
        s=s,
        c=C,
        tau=TAU,
        tau_public=TAU_PUBLIC,
        sigma=SIGMA,
        theta=THETA,
        r=r,
        delta=delta,
        wrap_fn=wrap,
        label=label,
    )
    return content, r, eps, delta, account


def main(argv: list[str] | None = None) -> None:
    print("Starting WP2 pipeline")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-trajectories", type=int, default=N_TRAJECTORIES)
    parser.add_argument(
        "--entries-per-cycle",
        type=int,
        required=True,
        help="Public fixed entry count B used by every complete Step-1 cycle.",
    )
    parser.add_argument("--k-labels", type=int, default=K_LABELS)
    parser.add_argument("--x-per-label", type=int, default=X_PER_LABEL)
    parser.add_argument("--epsilon-per-round", type=float, default=EPSILON_PER_ROUND)
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--aim", default=AIM_DEFAULT)
    parser.add_argument("--url", default=URL_DEFAULT)
    parser.add_argument("--user-prefix", default=USER_PREFIX)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--model", choices=("qwen", "mistral"), default=MODEL_DEFAULT)
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument(
        "--skip-trajectories",
        action="store_true",
        help=(
            "Skip the WP1.5 trajectory loop and consume whichever per-user "
            "stores already exist under --memory-dir. Useful for re-running "
            "only the WP2 shared-memory pipeline on a fixed input."
        ),
    )
    args = parser.parse_args(argv)

    if args.n_trajectories < 1 and not args.skip_trajectories:
        parser.error("--n-trajectories must be >= 1 unless --skip-trajectories is set.")
    if args.k_labels < 1:
        parser.error("--k-labels must be >= 1.")
    if args.entries_per_cycle < 1:
        parser.error("--entries-per-cycle must be >= 1.")
    if args.x_per_label < 1:
        parser.error("--x-per-label must be >= 1.")
    if not 0.0 < args.delta <= 1.0:
        parser.error("--delta must be in (0, 1].")
    if args.x_per_label == 1:
        print(
            "[WP2] WARNING: x_per_label=1 always triggers round 2; "
            "the per-trigger leak bound does not apply."
        )

    memory_dir = args.memory_dir
    shared_path = memory_dir / SHARED_STORE_FILENAME
    buffer_path = memory_dir / BUFFER_FILENAME
    checkpoint_path = memory_dir / CHECKPOINT_FILENAME
    audit_path = memory_dir / AUDIT_FILENAME

    load_dotenv()
    seed = load_random_seed()

    if not args.skip_trajectories:
        _run_trajectories(
            n=args.n_trajectories,
            aim=args.aim,
            url=args.url,
            user_prefix=args.user_prefix,
            memory_dir=memory_dir,
            max_steps=args.max_steps,
            headless=args.headless,
            model=args.model,
        )
        print("Ran all agent trajectories")
    else:
        print("[WP2] --skip-trajectories: re-using existing per-user stores.")

    embedder = Embedder()
    checkpoint = _load_checkpoint(checkpoint_path)
    consumed_ids: set[str] = set(checkpoint["consumed_entry_ids"])
    step1_processed_ids: set[str] = set(checkpoint["step1_processed_entry_ids"])
    carry_over = read_buffer(buffer_path)

    step1_processed_ids.update(consumed_ids)
    new_entries, assignment_batch, step1_processed_ids = _assemble_cycle_entries(
        memory_dir,
        step1_processed_ids,
        carry_over,
        embedder,
        entries_per_cycle=args.entries_per_cycle,
    )
    print(
        f"[WP2] Checkpoint: trigger_count={checkpoint['trigger_count']}, "
        f"cumulative_epsilon={checkpoint['cumulative_epsilon']:.4f}, "
        f"cumulative_delta={checkpoint['cumulative_delta']:.3e}, "
        f"step1_processed_entry_ids={len(step1_processed_ids)}, "
        f"consumed_entry_ids={len(consumed_ids)}"
    )

    print(
        f"[WP2] New entries: {len(new_entries)}; carry-over: {len(carry_over)}; "
        f"round-2 assignment batch size: {len(assignment_batch)}"
    )

    if not assignment_batch:
        print(
            f"[WP2] Fewer than B={args.entries_per_cycle} new entries are available; "
            "leaving them pending and preserving carry-over."
        )
        return

    # K labels from the B new entries only. Carry-over already paid round 1.
    labels, label_flags, r_round1, eps_round1, delta_round1 = _run_round1(
        new_entries,
        expected_batch_size=args.entries_per_cycle,
        k_labels=args.k_labels,
        target_epsilon=args.epsilon_per_round,
        delta=args.delta,
    )
    step1_processed_ids.update(entry_id(entry) for entry in new_entries)

    # New + held entries, assigned onto this trigger's labels.
    assignments = assign_memories_to_labels(assignment_batch, labels, embedder=embedder)
    buckets = group_by_label(assignments, n_labels=args.k_labels)
    print(f"[WP2] Per-label bucket sizes: {[len(b) for b in buckets]}")

    gating = select_round2_inputs(buckets, x_per_label=args.x_per_label)
    print(
        f"[WP2] Triggered labels (bucket >= X={args.x_per_label}): "
        f"{gating.triggered_labels} of {args.k_labels}"
    )
    print(f"[WP2] Carry-over count: {len(gating.carry_over)}")

    shared_store = MemoryStore.load(shared_path, user_id="shared", embedder=embedder)
    ollama_client = OllamaClient(model=QWEN_MODEL, seed=seed)

    round2_records: list[dict] = []
    consumed_indices: set[int] = set()
    skipped_round2_indices: list[int] = []
    # Each triggered label's round-2 batch is a disjoint subset of the
    # assignment set (every entry goes to exactly one label). Parallel
    # composition (Amin Lemma 2): trigger-level round-2 cost is
    # max(eps_r2) across labels, not the sum. Collected here, reduced
    # after the loop.
    eps_round2_per_label: list[float] = []

    for label_idx in gating.triggered_labels:
        label_str = labels[label_idx]
        entry_indices = gating.label_inputs[label_idx]
        assert entry_indices is not None  # triggered_labels filters None out
        round2_entries = [assignment_batch[i] for i in entry_indices]
        result = _run_round2_for_label(
            label=label_str,
            entries=round2_entries,
            expected_batch_size=args.x_per_label,
            target_epsilon=args.epsilon_per_round,
            delta=args.delta,
        )
        if result is None:
            # Budget too tight for round 2. Round 1 is already paid;
            # park these in carry-over in case a later trigger retries.
            skipped_round2_indices.extend(entry_indices)
            continue
        content, r_r2, eps_r2, delta_r2, account = result
        eps_round2_per_label.append(eps_r2)
        consumed_indices.update(entry_indices)

        print(f"[WP2] Round 2 [{label_str!r}] DP content: {content!r}")
        print(
            f"      private tokens used = {account.private_tokens_used} / {account.r}, "
            f"public tokens used = {account.public_tokens_used}"
        )

        title, description = title_and_description(
            content=content,
            label=label_str,
            client=ollama_client,
        )
        print(f"[WP2] Post-processed title={title!r}")
        print(f"[WP2] Post-processed description={description!r}")

        item = MemoryItem(title=title, description=description, content=content)
        shared_store.add_entry(query=label_str, outcome="shared", items=[item])
        print(f"[WP2] Wrote shared entry under query={label_str!r}.")

        round2_records.append(
            {
                "label_idx": label_idx,
                "label": label_str,
                "S": args.x_per_label,
                "actual_batch_size": len(round2_entries),
                "r": r_r2,
                "delta": delta_r2,
                "eps": eps_r2,
                "title": title,
                "description": description,
                "content": content,
                "private_tokens_used": account.private_tokens_used,
                "public_tokens_used": account.public_tokens_used,
                "entry_ids": [entry_id(entry) for entry in round2_entries],
            }
        )

    carry_indices = list(gating.carry_over) + skipped_round2_indices
    carry_entries = [assignment_batch[i] for i in carry_indices]
    write_buffer(carry_entries, buffer_path)
    print(
        f"[WP2] Persisted {len(carry_entries)} entries to carry-over buffer "
        f"({len(gating.carry_over)} from gating + {len(skipped_round2_indices)} "
        f"from round-2 skips)."
    )

    for i in consumed_indices:
        consumed_ids.add(entry_id(assignment_batch[i]))

    # Parallel composition across disjoint per-label round-2 batches:
    # trigger-level round-2 cost is max(eps_r2), not the sum. The
    # cumulative audit adds every trigger's released mechanisms;
    # carry-over does not pay round 1 again.
    eps_round2_trigger = max(eps_round2_per_label) if eps_round2_per_label else 0.0
    delta_round2_trigger = args.delta if eps_round2_per_label else 0.0
    trigger_epsilon = eps_round1 + eps_round2_trigger
    trigger_delta = delta_round1 + delta_round2_trigger
    cumulative_epsilon = checkpoint["cumulative_epsilon"] + eps_round1 + eps_round2_trigger
    cumulative_delta = checkpoint["cumulative_delta"] + trigger_delta
    trigger_idx = checkpoint["trigger_count"] + 1
    new_checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "last_run_at": _utcnow_iso(),
        "step1_processed_entry_ids": sorted(step1_processed_ids),
        "consumed_entry_ids": sorted(consumed_ids),
        "trigger_count": trigger_idx,
        "cumulative_epsilon": cumulative_epsilon,
        "cumulative_delta": cumulative_delta,
    }
    _save_checkpoint(checkpoint_path, new_checkpoint)

    audit_record = {
        "trigger_idx": trigger_idx,
        "trigger_at": new_checkpoint["last_run_at"],
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "entries_per_cycle": args.entries_per_cycle,
        "k_labels": args.k_labels,
        "x_per_label": args.x_per_label,
        "s1": args.entries_per_cycle,
        "s3": args.x_per_label,
        "delta": args.delta,
        "round1": {
            "S": args.entries_per_cycle,
            "actual_batch_size": len(new_entries),
            "r": r_round1,
            "delta": delta_round1,
            "eps": eps_round1,
            "labels": labels,
            "label_parsed_flags": label_flags,
            "entry_ids": [entry_id(entry) for entry in new_entries],
            "new_entry_count": len(new_entries),
        },
        "round2_assignment": {
            "entry_count": len(assignment_batch),
            "new_entry_count": len(new_entries),
            "carry_over_entry_count": len(carry_over),
            "entry_ids": [entry_id(entry) for entry in assignment_batch],
            "bucket_sizes": [len(b) for b in buckets],
            "triggered_labels": gating.triggered_labels,
        },
        "round2_per_label": round2_records,
        "round2_eps_parallel_composed": eps_round2_trigger,
        "round2_delta_parallel_composed": delta_round2_trigger,
        "trigger_epsilon": trigger_epsilon,
        "trigger_delta": trigger_delta,
        "n_consumed": len(consumed_indices),
        "n_carryover_gating": len(gating.carry_over),
        "n_carryover_round2_skipped": len(skipped_round2_indices),
        "n_carryover_total": len(carry_entries),
        "carry_over_entry_ids": [entry_id(entry) for entry in carry_entries],
        "cumulative_epsilon": cumulative_epsilon,
        "cumulative_delta": cumulative_delta,
    }
    _append_audit(audit_path, audit_record)

    print()
    print("=" * 72)
    print(f"[WP2] Trigger {trigger_idx} complete.")
    print(f"      Round 1: S={args.entries_per_cycle}, eps={eps_round1:.4f}")
    print(
        f"      Round 2: {len(round2_records)} label(s) wrote shared entries; "
        f"eps_round2_trigger={eps_round2_trigger:.4f} "
        f"(parallel-composed max of per-label eps={[round(e, 4) for e in eps_round2_per_label]})"
    )
    print(f"      Consumed this trigger: {len(consumed_indices)} entries")
    print(f"      Carried over to next trigger: {len(carry_entries)} entries")
    print(f"      Cumulative epsilon: {cumulative_epsilon:.4f}")
    print(f"      Cumulative delta: {cumulative_delta:.3e}")
    print(f"      Audit log: {audit_path}")
    print(f"      Shared store: {shared_path} ({len(shared_store)} total entries)")


if __name__ == "__main__":
    main()
