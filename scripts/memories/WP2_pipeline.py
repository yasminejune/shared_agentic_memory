"""End-to-end WP2 cross-user shared-memory pipeline driver.

One script, top-level constants for every knob, one ``main()``.
Takes a task aim (e.g. ``"Find cheapest hairbrush on Amazon"``),
runs ``N_TRAJECTORIES`` private agent trajectories across distinct
simulated user_ids, then drives the two-round Amin et al. (2024)
shared-memory pipeline with item-level privacy, fixed public expected
batch sizes, and X-per-label gating.

The pipeline flow (consistent with the WP2-plan §2 architecture
diagram, with the two user-confirmed deviations recorded below):

1. **Trajectories.** Spawn ``N_TRAJECTORIES`` ``scripts/memories/WP1_5.py``
   subprocesses, each with its own ``--user-id`` so each writes to a
   distinct ``data/memories/<user_id>.jsonl``. ``--skip-trajectories``
   bypasses this step and re-uses whatever per-user stores are
   already on disk (useful for the smoke run and for re-running the
   shared-memory pipeline on a fixed input).

2. **Assemble cycle inputs.** Flatten every per-user trajectory entry
   into its 1-3 ``MemoryItem`` records and select exactly the public
   ``B = --items-per-cycle`` items whose item IDs are absent from
   ``data/memories/.shared_state.json#step1_processed_item_ids``.
   Separately load the carry-over buffer from
   ``data/memories/.shared_buffer.jsonl``. Carry-over items have
   already paid their round-1 privacy cost and therefore rejoin the
   pipeline at round-2 batch assignment, not at round 1.

3. **Round 1 (Amin Algorithm 1, S = B).** Each ``MemoryItem`` is one
   protected example and one prompt row. ``S`` is the public fixed
   cycle size, ``delta`` is fixed explicitly, and ``r`` is solved
   against ``EPSILON_PER_ROUND``. Each item's parent trajectory
   query is rendered via :func:`_render_query_block`. The DP-released
   labels become the ``query`` field (and hence the embedding
   anchor) of every shared memory; feeding queries keeps those
   labels in the same semantic space as the future task aims that
   retrieval will embed against (WP2-plan §7.1). The mechanism then
   samples ``K_LABELS`` topic labels via
   :func:`agent_memories.agent.privacy.generate` with
   ``wrap_fn=wrap_label``. The JSON output is parsed via
   :func:`parse_json_labels` (the production
   :data:`~agent_memories.agent.privacy.prompts.LABEL_PROMPT` is the
   JSON template); missing slots become ``label_<i>``.

4. **Round-2 batch assignment.** Embed the ``K`` new labels through the
   same :class:`~agent_memories.memory.Embedder` the WP1.6 stores
   use, then call :func:`assign_memories_to_labels` to bucket every
   new and carry-over item under its nearest new label by cosine
   similarity. The label count remains exactly ``K``; only the
   number of items assigned to each bucket varies. WP2-plan §4.3
   (the WP2.5 resolution) justifies that this satisfies Amin
   Assumption 1: each prompt's bucket depends only on the prompt
   itself and the DP-released labels (public via post-processing).

5. **X-gating.** Per the user's confirmed ``keep_and_document``
   choice, the gating threshold ``X_PER_LABEL > 1`` is honoured
   even though it introduces content-dependent gating (WP2-plan §10
   open question 2). Every qualifying bucket is passed to round 2 in
   full; only buckets below ``X`` are pushed to the carry-over buffer
   for the next trigger.

6. **Round 2 (Amin Algorithm 1, expected S = X).** Per qualifying
   label, one Amin run receives every item as a separate prompt row while
   retaining ``S = X_PER_LABEL`` as the expected batch size,
   using the same fixed ``delta`` as round 1, and ``r`` solved against
   ``EPSILON_PER_ROUND``. Amin et al. permit actual batch sizes to
   differ from expected ``S`` without invalidating Theorem 1. The
   wrap function is :func:`agent_memories.agent.privacy.prompts.wrap`
   with the DP-released label as ``label=...``; the output is the
   ``content`` field of a single shared :class:`MemoryItem`.

7. **Post-processing (off-DP, free).**
   :func:`agent_memories.generalisation.title_and_description`
   turns ``(content, label)`` into the matching ``title`` and
   ``description`` via Qwen / Ollama. By the post-processing
   property of differential privacy this call contributes zero to
   ``rho_total`` (WP2-plan §4.7).

8. **Shared store write.** The completed
   :class:`MemoryItem(title, description, content)` is appended to
   ``data/memories/shared.jsonl`` via the shared
   :class:`MemoryStore` (``user_id="shared"``, ``query=<label>``,
   ``outcome="shared"``).

9. **Persist buffer + checkpoint + audit log.** The carry-over
   buffer is overwritten with this trigger's "set aside" items;
   the checkpoint records which items have completed round 1,
   separately records which items have contributed to a shared
   memory, and accumulates ``cumulative_epsilon`` across triggers;
   one line is appended to ``data/memories/.shared_audit.jsonl``
   with the per-round ``(S, r, eps)``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent_memories.agent.privacy import (
    PrivacyAccount,
    epsilon_from_rho,
    generate,
    rho_for,
    solve_r,
)
from agent_memories.agent.privacy.prompts import wrap, wrap_label
from agent_memories.config import DEFAULT_MEMORY_DIR, load_random_seed
from agent_memories.generalisation import (
    CycleItem,
    assign_memories_to_labels,
    flatten_entry,
    group_by_label,
    parse_json_labels,
    read_buffer,
    select_round2_inputs,
    title_and_description,
    write_buffer,
)
from agent_memories.memory import Embedder, MemoryItem, MemoryStore
from agent_memories.services.ollama_client import OllamaClient

REPO_ROOT = Path(__file__).resolve().parents[2]
WP1_5_SCRIPT = REPO_ROOT / "scripts" / "memories" / "WP1_5.py"

# ------------------------------------------------------------------
# Trigger-level config (overridable via CLI flags below)
# ------------------------------------------------------------------
N_TRAJECTORIES = 5
K_LABELS = 3  # Number of new labels to generate per trigger
X_PER_LABEL = 5  # Number of items required for a round-2 shared memory generation

AIM_DEFAULT = "Find cheapest hairbrush on Amazon"
URL_DEFAULT = "https://www.amazon.co.uk"
USER_PREFIX = "user_"
MAX_STEPS_DEFAULT = 10
MODEL_DEFAULT = "qwen"

QWEN_MODEL = "qwen3.5:4b-nvfp4"

# ------------------------------------------------------------------
# Amin Algorithm 1 hyperparameters (one set, reused for both rounds)
# ------------------------------------------------------------------
C = 50.0
TAU = 1.0
TAU_PUBLIC = 1.5
SIGMA = 0.1
THETA = 0.0
R_MAX = 80
EPSILON_PER_ROUND = 200.0
DELTA = 1e-5

# ------------------------------------------------------------------
# Filesystem layout
# ------------------------------------------------------------------
SHARED_STORE_FILENAME = "shared.jsonl"
BUFFER_FILENAME = ".shared_buffer.jsonl"
CHECKPOINT_FILENAME = ".shared_state.json"
AUDIT_FILENAME = ".shared_audit.jsonl"
CHECKPOINT_SCHEMA_VERSION = 2


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _render_item_block(item: CycleItem) -> str:
    """Render one protected memory item for a round-2 prompt row."""
    return f"{item.item.title} | {item.item.description} | {item.item.content}"


def _render_query_block(item: CycleItem) -> str:
    """Render one item's parent query as its round-1 prompt row.

    Round 1 synthesises the DP-released labels that become the
    ``query`` field (and hence the embedding anchor) of every shared
    memory written to ``data/memories/shared.jsonl``. Feeding the
    per-trajectory task aim -- rather than the reasoning-step items
    block -- keeps those labels in the same semantic space as the
    task aims that future agent queries will embed against, which is
    the retrieval direction the shared store is designed for
    (WP2-plan §7.1, ``store.py`` ``MemoryStore.add_entry``). Round 2
    still uses :func:`_render_item_block` because its output is the
    distilled ``content`` field, which needs the concrete reasoning
    steps.
    """
    return item.query


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

    Each trajectory writes to ``data/memories/<user_id>.jsonl``. A
    non-zero exit code from any one trajectory is logged but does
    not abort the loop -- the pipeline still tries to consume
    whichever per-user stores are non-empty afterwards.
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
            "step1_processed_item_ids": [],
            "consumed_item_ids": [],
            "trigger_count": 0,
            "cumulative_epsilon": 0.0,
            "cumulative_delta": 0.0,
        }
    state: dict = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            "Checkpoint uses the legacy entry-level schema. Reset "
            ".shared_state.json and .shared_buffer.jsonl before running "
            "item-level privacy."
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


def _load_unprocessed_items(
    memory_dir: Path,
    step1_processed_ids: set[str],
    embedder: Embedder,
) -> list[CycleItem]:
    """Return item records that have not yet completed round 1.

    Only ``<memory_dir>/user_*.jsonl`` is considered; the reserved
    ``shared.jsonl`` (WP2.4 cross-user store) and any hidden file
    (e.g. ``.shared_buffer.jsonl``) are excluded so the WP2 input
    set is exactly "private per-user new memories".
    """
    new_items: list[CycleItem] = []
    for path in sorted(memory_dir.glob("user_*.jsonl")):
        store = MemoryStore.load(path, user_id=path.stem, embedder=embedder)
        for entry in store.all():
            for item in flatten_entry(entry):
                if item.item_id not in step1_processed_ids:
                    new_items.append(item)
    return new_items


def _assemble_cycle_items(
    memory_dir: Path,
    step1_processed_ids: set[str],
    carry_over: list[CycleItem],
    embedder: Embedder,
    *,
    items_per_cycle: int,
) -> tuple[list[CycleItem], list[CycleItem], set[str]]:
    """Return one complete round-1 item batch and round-2 candidates.

    Carry-over items prove that they completed round 1 in an earlier
    trigger. Adding their IDs before loading the user stores prevents the
    same protected examples from appearing once as new and once as carry-over.
    Fewer than ``items_per_cycle`` new examples remain pending in their
    source stores; no partial round-1 cycle is released.
    """
    processed_ids = set(step1_processed_ids)
    processed_ids.update(item.item_id for item in carry_over)
    available = _load_unprocessed_items(memory_dir, processed_ids, embedder)
    if len(available) < items_per_cycle:
        return [], [], processed_ids
    new_items = available[:items_per_cycle]
    return new_items, new_items + carry_over, processed_ids


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
    round1_batch: list[CycleItem],
    *,
    expected_batch_size: int,
    k_labels: int,
    target_epsilon: float,
    delta: float,
) -> tuple[list[str], list[bool], int, float, float]:
    """Run round 1 once; return labels, parsed flags, r, eps, delta.

    Every item is one prompt row. Accounting uses the public fixed
    cycle size rather than deriving ``s`` or ``delta`` from private
    realised counts.
    """
    s = expected_batch_size
    if len(round1_batch) != s:
        raise ValueError(f"Round 1 requires exactly B={s} items; got {len(round1_batch)}.")
    r = solve_r(target_epsilon, delta, s=s, c=C, tau=TAU, sigma=SIGMA, r_max=R_MAX)
    if r == 0:
        raise RuntimeError(
            f"Round 1 budget too tight: target_epsilon={target_epsilon}, "
            f"delta={delta}, S={s} -> r=0. Loosen EPSILON_PER_ROUND or reduce sigma."
        )
    eps = epsilon_from_rho(rho_for(r, s, C, TAU, SIGMA), delta)
    _print_round_summary("Round 1", s=s, delta=delta, r=r, eps=eps, target_epsilon=target_epsilon)

    items_blocks = [_render_query_block(item) for item in round1_batch]
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
    items: list[CycleItem],
    expected_batch_size: int,
    target_epsilon: float,
    delta: float,
) -> tuple[str, int, float, float, PrivacyAccount] | None:
    """Run one round-2 generate call; return DP content + privacy account.

    Every item assigned to the qualifying label is included. Privacy
    accounting continues to use the fixed threshold as Amin et al.'s
    expected batch size, which need not equal the actual batch size.

    Returns ``None`` when ``solve_r`` cannot fit even one private
    token inside the budget (so the caller can skip the post-process
    step for this label without aborting the whole trigger).
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

    items_blocks = [_render_item_block(item) for item in items]
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
        "--items-per-cycle",
        type=int,
        required=True,
        help="Public fixed item count B used by every complete Step-1 cycle.",
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
    if args.items_per_cycle < 1:
        parser.error("--items-per-cycle must be >= 1.")
    if args.x_per_label < 1:
        parser.error("--x-per-label must be >= 1.")
    if not 0.0 < args.delta <= 1.0:
        parser.error("--delta must be in (0, 1].")
    if args.x_per_label == 1:
        print(
            "[WP2] WARNING: x_per_label=1 collapses the X-gating to 'always trigger'; "
            "the per-trigger leak the user opted into (WP2-plan §10 OQ 2) does not "
            "apply in this regime."
        )

    memory_dir = args.memory_dir
    shared_path = memory_dir / SHARED_STORE_FILENAME
    buffer_path = memory_dir / BUFFER_FILENAME
    checkpoint_path = memory_dir / CHECKPOINT_FILENAME
    audit_path = memory_dir / AUDIT_FILENAME

    load_dotenv()
    seed = load_random_seed()

    # Run the agent trajectories to get the new entries
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

    # Flatten private entries into item-level protected examples.
    embedder = Embedder()
    checkpoint = _load_checkpoint(checkpoint_path)
    consumed_ids: set[str] = set(checkpoint["consumed_item_ids"])
    step1_processed_ids: set[str] = set(checkpoint["step1_processed_item_ids"])
    carry_over = read_buffer(buffer_path)

    step1_processed_ids.update(consumed_ids)
    new_items, assignment_batch, step1_processed_ids = _assemble_cycle_items(
        memory_dir,
        step1_processed_ids,
        carry_over,
        embedder,
        items_per_cycle=args.items_per_cycle,
    )
    print(
        f"[WP2] Checkpoint: trigger_count={checkpoint['trigger_count']}, "
        f"cumulative_epsilon={checkpoint['cumulative_epsilon']:.4f}, "
        f"cumulative_delta={checkpoint['cumulative_delta']:.3e}, "
        f"step1_processed_item_ids={len(step1_processed_ids)}, "
        f"consumed_item_ids={len(consumed_ids)}"
    )

    print(
        f"[WP2] New items: {len(new_items)}; carry-over: {len(carry_over)}; "
        f"round-2 assignment batch size: {len(assignment_batch)}"
    )

    if not assignment_batch:
        print(
            f"[WP2] Fewer than B={args.items_per_cycle} new items are available; "
            "leaving them pending and preserving carry-over."
        )
        return

    # Generate exactly K labels from B new item examples only. Carry-over
    # items already paid this privacy cost in an earlier trigger.
    labels, label_flags, r_round1, eps_round1, delta_round1 = _run_round1(
        new_items,
        expected_batch_size=args.items_per_cycle,
        k_labels=args.k_labels,
        target_epsilon=args.epsilon_per_round,
        delta=args.delta,
    )
    step1_processed_ids.update(item.item_id for item in new_items)

    # Reassign both new and held items to the newly generated labels.
    assignments = assign_memories_to_labels(assignment_batch, labels, embedder=embedder)
    buckets = group_by_label(assignments, n_labels=args.k_labels)
    print(f"[WP2] Per-label bucket sizes: {[len(b) for b in buckets]}")

    # Only generate a shared memory if there are sufficient bundles for the label
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
    # assignment dataset (the assignment step routes every item to exactly
    # one label), so by parallel composition (Amin Lemma 2; WP2-plan
    # §2 line 30 and §4.1 line 169) the round-2 cost for this trigger is
    # max(eps_r2) across labels, not the sum. The per-label values are
    # collected here and reduced once the loop ends.
    eps_round2_per_label: list[float] = []

    # Run round 2 of Amin et al to generate the shared memory for the triggered labels
    # In a second step, generate the title and description for the shared memory
    for label_idx in gating.triggered_labels:
        label_str = labels[label_idx]
        item_indices = gating.label_inputs[label_idx]
        assert item_indices is not None  # triggered_labels filters None out
        round2_items = [assignment_batch[i] for i in item_indices]
        result = _run_round2_for_label(
            label=label_str,
            items=round2_items,
            expected_batch_size=args.x_per_label,
            target_epsilon=args.epsilon_per_round,
            delta=args.delta,
        )
        if result is None:
            # Round 2 was skipped for budget reasons; the items already
            # paid the round-1 cost but did not contribute to a shared
            # memory. Push them to the carry-over buffer so a future
            # trigger can retry them if the privacy configuration changes.
            skipped_round2_indices.extend(item_indices)
            continue
        content, r_r2, eps_r2, delta_r2, account = result
        eps_round2_per_label.append(eps_r2)
        consumed_indices.update(item_indices)

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
                "actual_batch_size": len(round2_items),
                "r": r_r2,
                "delta": delta_r2,
                "eps": eps_r2,
                "title": title,
                "description": description,
                "content": content,
                "private_tokens_used": account.private_tokens_used,
                "public_tokens_used": account.public_tokens_used,
                "item_ids": [item.item_id for item in round2_items],
            }
        )

    carry_indices = list(gating.carry_over) + skipped_round2_indices
    carry_items = [assignment_batch[i] for i in carry_indices]
    write_buffer(carry_items, buffer_path)
    print(
        f"[WP2] Persisted {len(carry_items)} item(s) to carry-over buffer "
        f"({len(gating.carry_over)} from gating + {len(skipped_round2_indices)} "
        f"from round-2 skips)."
    )

    for i in consumed_indices:
        consumed_ids.add(assignment_batch[i].item_id)

    # Parallel composition across the disjoint per-label round-2 batches:
    # the trigger-level round-2 cost is max(eps_r2), not sum (WP2-plan §4.1).
    # The cumulative audit value conservatively adds every trigger's released
    # mechanisms; carry-over memories do not pay round-1 cost again.
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
        "step1_processed_item_ids": sorted(step1_processed_ids),
        "consumed_item_ids": sorted(consumed_ids),
        "trigger_count": trigger_idx,
        "cumulative_epsilon": cumulative_epsilon,
        "cumulative_delta": cumulative_delta,
    }
    _save_checkpoint(checkpoint_path, new_checkpoint)

    audit_record = {
        "trigger_idx": trigger_idx,
        "trigger_at": new_checkpoint["last_run_at"],
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "items_per_cycle": args.items_per_cycle,
        "k_labels": args.k_labels,
        "x_per_label": args.x_per_label,
        "s1": args.items_per_cycle,
        "s3": args.x_per_label,
        "delta": args.delta,
        "round1": {
            "S": args.items_per_cycle,
            "actual_batch_size": len(new_items),
            "r": r_round1,
            "delta": delta_round1,
            "eps": eps_round1,
            "labels": labels,
            "label_parsed_flags": label_flags,
            "item_ids": [item.item_id for item in new_items],
            "new_item_count": len(new_items),
        },
        "round2_assignment": {
            "item_count": len(assignment_batch),
            "new_item_count": len(new_items),
            "carry_over_item_count": len(carry_over),
            "item_ids": [item.item_id for item in assignment_batch],
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
        "n_carryover_total": len(carry_items),
        "carry_over_item_ids": [item.item_id for item in carry_items],
        "cumulative_epsilon": cumulative_epsilon,
        "cumulative_delta": cumulative_delta,
    }
    _append_audit(audit_path, audit_record)

    print()
    print("=" * 72)
    print(f"[WP2] Trigger {trigger_idx} complete.")
    print(f"      Round 1: S={args.items_per_cycle}, eps={eps_round1:.4f}")
    print(
        f"      Round 2: {len(round2_records)} label(s) wrote shared entries; "
        f"eps_round2_trigger={eps_round2_trigger:.4f} "
        f"(parallel-composed max of per-label eps={[round(e, 4) for e in eps_round2_per_label]})"
    )
    print(f"      Consumed this trigger: {len(consumed_indices)} item(s)")
    print(f"      Carried over to next trigger: {len(carry_items)} item(s)")
    print(f"      Cumulative epsilon: {cumulative_epsilon:.4f}")
    print(f"      Cumulative delta: {cumulative_delta:.3e}")
    print(f"      Audit log: {audit_path}")
    print(f"      Shared store: {shared_path} ({len(shared_store)} total entries)")


if __name__ == "__main__":
    main()
