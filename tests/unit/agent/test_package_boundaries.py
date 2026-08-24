"""Guards on the agent/ package split: lm, amin_et_al, invisible_ink.

agent/lm holds the Gemma wrapper and the prompt templates that both DP
mechanisms build on. Each mechanism depends on lm and on nothing else in
agent/, so neither one can grow an implicit dependency on the other. The
imports are read off the AST rather than executed, so these tests never
load torch or transformers.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

AGENT_DIR = Path(__file__).resolve().parents[3] / "src" / "agent_memories" / "agent"


def _imported_agent_packages(package: str) -> set[str]:
    """Sibling packages under agent/ that `package` imports, absolute or relative.

    A relative import is resolved by level: inside agent/<package>/, level 2
    means "up to agent/", so the first name in the module path is the sibling.
    """
    found: set[str] = set()
    for path in sorted((AGENT_DIR / package).glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module:
                found.add(node.module.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found |= _sibling_from_absolute(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    found |= _sibling_from_absolute(alias.name)
    return found - {package}


def _sibling_from_absolute(dotted: str) -> set[str]:
    """The agent/ sibling named by a fully qualified module path, if any."""
    parts = dotted.split(".")
    if parts[:2] == ["agent_memories", "agent"] and len(parts) > 2:
        return {parts[2]}
    return set()


def test_lm_depends_on_neither_mechanism() -> None:
    """The shared layer must stay underneath both mechanisms."""
    assert _imported_agent_packages("lm") & {"amin_et_al", "invisible_ink"} == set()


def test_amin_does_not_import_invisible_ink() -> None:
    assert "invisible_ink" not in _imported_agent_packages("amin_et_al")


def test_invisible_ink_does_not_import_amin() -> None:
    """InvisibleInk is not built on Amin et al.; it only shares lm."""
    assert "amin_et_al" not in _imported_agent_packages("invisible_ink")


def test_both_mechanisms_go_through_lm() -> None:
    for package in ("amin_et_al", "invisible_ink"):
        assert "lm" in _imported_agent_packages(package), f"{package} should build on lm"


def test_privacy_package_is_gone() -> None:
    """The old catch-all package must not come back."""
    assert not (AGENT_DIR / "privacy").exists()


def test_amin_public_api_is_intact() -> None:
    """The names the WP2 scripts import off the package."""
    from agent_memories.agent import amin_et_al

    expected = {
        "DeltaCheck",
        "PrivacyAccount",
        "check_delta",
        "delta_from_rho_epsilon",
        "epsilon_from_rho",
        "generate",
        "rho_for",
        "solve_r",
    }
    assert expected <= set(amin_et_al.__all__)
    for name in expected:
        assert hasattr(amin_et_al, name)


def test_prompts_import_does_not_pull_in_torch() -> None:
    """Splitting prompts out of the mechanisms keeps them cheap to import.

    generalisation/ wants the templates without the sampler, so a fresh
    interpreter importing prompts must not end up with torch loaded.
    """
    code = (
        "import sys; import agent_memories.agent.lm.prompts; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_wrap_helpers_are_reachable_from_lm() -> None:
    from agent_memories.agent.lm.prompts import wrap, wrap_label

    assert "topic" in wrap("a", label="topic")
    assert "3" in wrap_label("a", k=3)
