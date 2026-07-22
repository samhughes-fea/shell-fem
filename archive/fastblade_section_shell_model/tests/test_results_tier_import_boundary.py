"""Import-boundary guard: each results tier imports only the tier above it.

Per PROCESSING_ARCHITECTURE_PROPOSAL.md's strict consumption rule: tier-(n)
consumes only tier-(n-1)'s result containers (+ containers/cache, which are
shared infrastructure, not a "tier"). Concretely:

- ``compute_secondary`` (tier 2) must not import from ``compute_tertiary`` or
  ``compute_quaternary`` at module scope.
- ``compute_tertiary`` (tier 3) must not import from ``compute_quaternary`` at
  module scope.
- ``compute_quaternary`` (tier 4) must not import from ``compute_secondary``
  (it may import ``compute_tertiary`` — the tier immediately above it — e.g.
  provenance gating helpers).
"""

from __future__ import annotations

import ast
from pathlib import Path

_RESULTS = Path(__file__).resolve().parents[1] / "processing" / "results"


def _module_level_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in tree.body:  # top level only — skip anything nested in functions/classes
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append("." * node.level + node.module)
    return imports


def _imports_tier(imports: list[str], tier_pkg: str) -> bool:
    return any(tier_pkg in imp for imp in imports)


def _check_tier_does_not_import(tier_dir: str, forbidden_tiers: tuple[str, ...]) -> None:
    root = _RESULTS / tier_dir
    assert root.is_dir(), root
    violations: list[str] = []
    for path in root.rglob("*.py"):
        imports = _module_level_imports(path)
        for forbidden in forbidden_tiers:
            if _imports_tier(imports, forbidden):
                violations.append(f"{path.relative_to(_RESULTS)}: imports {forbidden} at module scope")
    assert not violations, "\n".join(violations)


def test_compute_secondary_does_not_import_tier3_or_tier4():
    _check_tier_does_not_import("compute_secondary", ("compute_tertiary", "compute_quaternary"))


def test_compute_tertiary_does_not_import_tier4_at_module_scope():
    _check_tier_does_not_import("compute_tertiary", ("compute_quaternary",))


def test_compute_quaternary_does_not_import_tier2():
    _check_tier_does_not_import("compute_quaternary", ("compute_secondary",))


def test_compute_quaternary_hashin_module_has_no_failure_criterion_logic_in_tertiary():
    """Tier 3 hosts CLT recovery only; no ``hashin_fi``/failure-criterion calls."""
    tertiary_src = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (_RESULTS / "compute_tertiary").rglob("*.py")
    )
    assert "hashin_fi(" not in tertiary_src
    assert "materials.failure_criteria" not in tertiary_src
