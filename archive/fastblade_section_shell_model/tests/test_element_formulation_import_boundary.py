"""Import-boundary guard: engine/processing must not assemble via formulation kernels directly."""

from __future__ import annotations

import ast
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
_ENGINE = _PKG / "engine"
_PROCESSING = _PKG / "processing"
_OPERATIONS = _PROCESSING / "operations"
_FORBIDDEN_MODULES = (
    "element_library.mitc4.element",
)
# Registry / recovery shims until full recovery migration off mitc4.element helpers.
_ALLOWLIST = {
    "engine/global_shell_assembly.py",  # post-solve recovery only
    "engine/shell_solve_registry.py",
    "engine/station_theory_dispatch.py",
    "processing/results/compute_secondary/secondary_results_orchestrator.py",
    "processing/results/compute_tertiary/tertiary_results_orchestrator.py",
}


def _iter_py_files(root: Path):
    for path in root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        yield path


def _file_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _normalize(mod: str) -> str:
    return mod.replace("blade_precompute.section_shell_model.", "")


def _check_no_kernel_imports(root: Path, label: str) -> None:
    violations: list[str] = []
    for path in _iter_py_files(root):
        rel = path.relative_to(_PKG).as_posix()
        if rel in _ALLOWLIST:
            continue
        for mod in _file_imports(path):
            norm = _normalize(mod)
            for forbidden in _FORBIDDEN_MODULES:
                if norm == forbidden or norm.endswith(f".{forbidden}"):
                    violations.append(f"{rel}: import {mod}")
    assert not violations, f"{label} imports formulation kernels directly:\n" + "\n".join(violations)


def test_engine_does_not_import_formulation_kernel_modules():
    _check_no_kernel_imports(_ENGINE, "Engine")


def test_processing_does_not_import_formulation_kernel_modules():
    assert _PROCESSING.is_dir()
    _check_no_kernel_imports(_PROCESSING, "processing/")


def test_operations_has_no_orchestration_module():
    assert _OPERATIONS.is_dir()
    assert not (_OPERATIONS / "orchestrate.py").exists()
    ops_py = [p.name for p in _OPERATIONS.glob("*.py") if p.name != "__init__.py"]
    assert "recover_primary" not in ops_py


def test_global_shell_assembly_uses_formulation_cache_pipeline():
    src = (_PKG / "engine" / "global_shell_assembly.py").read_text(encoding="utf-8")
    assert "mitc4_stiffness(" not in src
    assert "assemble_section_from_panels(" in src
    thin = (_PKG / "engine" / "section_assembly.py").read_text(encoding="utf-8")
    assert "build_formulation_cache(" in thin
    assert "assemble_section_system(" in thin
    # R8: scatter-add implementation lives in processing/operations
    ops = (_PKG / "processing" / "operations" / "assemble_global.py").read_text(encoding="utf-8")
    assert "def assemble_section_system(" in ops
    assert "from ...engine.section_assembly import assemble_section_system" not in ops


def test_connected_assembly_core_has_no_legacy_solve_adapter():
    """``reduce_propagate_solve`` thin adapter was removed (Wave 3); solve via runner."""
    src = (_PKG / "engine" / "connected_assembly_core.py").read_text(encoding="utf-8")
    assert "reduce_propagate_solve" not in src
    assert "SectionLinearStaticRunner" not in src
    runner_src = (_PKG / "simulation_runner" / "section_linear_static.py").read_text(encoding="utf-8")
    assert "class SectionLinearStaticRunner" in runner_src


def test_engine_section_solve_shim_removed():
    """``engine/section_solve/`` was a one-cycle re-export shim (Phase 5 Wave A);
    it had zero inbound importers and is now deleted. Stage classes live in
    ``processing/operations/`` (see ``test_operations_has_no_orchestration_module``)."""
    assert not (_PKG / "engine" / "section_solve").exists()
    assert _OPERATIONS.is_dir()


def test_simulation_runner_owns_orchestration():
    src = (_PKG / "simulation_runner" / "section_linear_static.py").read_text(encoding="utf-8")
    assert "def prepare_local_system" in src
    assert "def solve_linear_system_only" in src
    assert "def compute_primary_results" in src
