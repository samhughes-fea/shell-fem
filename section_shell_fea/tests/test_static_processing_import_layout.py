"""Import compatibility and dependency-cycle guards for static processing families."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1]
_STATIC = _PKG / "processing" / "static"
_OPS = _STATIC / "operations"
_DIAG = _STATIC / "diagnostics"

_OPS_PAIRS = [
    ("contracts.solve_states", "solve_states", "GlobalShellSystem"),
    ("contracts.physics_contract", "physics_contract", "LinearFormulationPhysics"),
    ("preparation.prepare_local", "prepare_local", "PrepareLocalShellSystem"),
    ("assembly.assemble_global", "assemble_global", "AssembleGlobalShellSystem"),
    ("assembly.section_assembly", "section_assembly", "assemble_section_system"),
    ("reduction.constrain_global", "constrain_global", "ConstrainGlobalShellSystem"),
    ("reduction.reduce_constrained", "reduce_constrained", "ReduceConstrainedShellSystem"),
    ("solution.solve_reduced", "solve_reduced", "SolveReducedShellSystem"),
    ("solution.solve_constrained", "solve_constrained", "solve_constrained_system"),
    ("recovery.reconstruct_field", "reconstruct_field", "ReconstructGlobalShellField"),
    ("recovery.disassemble_field", "disassemble_field", "DisassembleGlobalShellField"),
]

_DIAG_PAIRS = [
    ("solve.solve_quality", "solve_quality", "assess_solve_quality"),
    ("solve.solve_phase_timing", "solve_phase_timing", "SolvePhaseCollector"),
    ("solve.dof_provenance", "dof_provenance", "dump_index_maps"),
    ("equilibrium.equilibrium_checks", "equilibrium_checks", "check_panel_equilibrium"),
    ("convergence.asme_gci", "asme_gci", "gci_triplet"),
    ("convergence.cascade_mesh_convergence", "cascade_mesh_convergence", "CascadeScalars"),
    ("convergence.grid_independence_study", "grid_independence_study", "run_grid_independence_study"),
    ("convergence.stability_scorecard", "stability_scorecard", "build_stability_scorecard"),
    ("parity.k6_parity", "k6_parity", "compare_strip_shell_k6"),
    ("parity.path_compare", "path_compare", "compare_path_resultants"),
    ("plots.mesh_plots", "mesh_plots", "save_section_centerline_mesh_figure"),
]


def _mod(suffix: str):
    return importlib.import_module(
        f"blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.{suffix}"
    )


@pytest.mark.parametrize("canonical,legacy,symbol", _OPS_PAIRS)
def test_operations_legacy_and_canonical_resolve_same_object(canonical, legacy, symbol):
    c = getattr(_mod(f"operations.{canonical}"), symbol)
    l = getattr(_mod(f"operations.{legacy}"), symbol)
    assert c is l


@pytest.mark.parametrize("canonical,legacy,symbol", _DIAG_PAIRS)
def test_diagnostics_legacy_and_canonical_resolve_same_object(canonical, legacy, symbol):
    c = getattr(_mod(f"diagnostics.{canonical}"), symbol)
    l = getattr(_mod(f"diagnostics.{legacy}"), symbol)
    assert c is l


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def _assert_no_forbidden_imports(root: Path, forbidden_substrings: tuple[str, ...], label: str) -> None:
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        # Skip root-level compatibility shims — they only re-export.
        if path.parent == root and "Compatibility shim" in path.read_text(encoding="utf-8")[:200]:
            continue
        for mod in _imported_modules(path):
            for bad in forbidden_substrings:
                if bad in mod:
                    violations.append(f"{path.relative_to(_PKG).as_posix()}: {mod}")
    assert not violations, f"{label} forbid imports:\n" + "\n".join(violations)


def test_operations_do_not_import_runner_or_convergence():
    _assert_no_forbidden_imports(
        _OPS,
        (
            "section_linear_static_runner",
            "diagnostics.convergence",
            "diagnostics.cascade_mesh_convergence",
            "diagnostics.grid_independence_study",
        ),
        "operations",
    )


def test_diagnostics_solve_and_equilibrium_do_not_import_runner_or_convergence():
    for family in ("solve", "equilibrium"):
        _assert_no_forbidden_imports(
            _DIAG / family,
            (
                "section_linear_static_runner",
                "diagnostics.convergence",
                "cascade_mesh_convergence",
                "grid_independence_study",
            ),
            f"diagnostics.{family}",
        )


def test_operation_family_packages_exist():
    for name in ("contracts", "preparation", "assembly", "reduction", "solution", "recovery"):
        assert (_OPS / name / "__init__.py").is_file()


def test_diagnostic_family_packages_exist():
    for name in ("solve", "equilibrium", "convergence", "parity", "plots"):
        assert (_DIAG / name / "__init__.py").is_file()
