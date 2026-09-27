"""Import-boundary guard: processing must not assemble via formulation kernels directly."""

from __future__ import annotations

import ast
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
_PROCESSING = _PKG / "processing"
_OPERATIONS = _PROCESSING / "static" / "operations"
_FORBIDDEN_MODULES = (
    "element_library.mitc4.element",
)
# Registry / recovery shims until full recovery migration off mitc4.element helpers.
# Buckling ops (assemble_geometric / solve_eigen) route via resolve_kernel — not allowlisted.
_ALLOWLIST: set[str] = set()


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
    return mod.replace("blade_structure_utilities.fem_utilities.section_shell_fea.", "")


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


def test_processing_does_not_import_formulation_kernel_modules():
    assert _PROCESSING.is_dir()
    _check_no_kernel_imports(_PROCESSING, "processing/")


def test_coordinate_frames_does_not_import_section_shell_fea():
    frames = (
        _PKG.parents[1] / "geometry_utilities" / "coordinate_frames"
    )
    violations: list[str] = []
    for path in frames.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "section_shell_fea" in text:
            violations.append(str(path.relative_to(frames)))
    assert not violations, "coordinate_frames must not import section_shell_fea:\n" + "\n".join(
        violations
    )


def test_formulation_factory_has_no_eager_mitc4_import():
    src = (_PKG / "pre_processing" / "element_library" / "formulation_factory.py").read_text(
        encoding="utf-8"
    )
    assert "from .mitc4.mitc4_quad4_element import" not in src
    assert "from .mitc4.factory import" in src or "mitc4.factory" in src



def test_operations_has_no_orchestration_module():
    assert _OPERATIONS.is_dir()
    assert not (_OPERATIONS / "orchestrate.py").exists()
    ops_py = [p.name for p in _OPERATIONS.glob("*.py") if p.name != "__init__.py"]
    assert "recover_primary" not in ops_py


def test_global_shell_assembly_uses_formulation_cache_pipeline():
    src = (_OPERATIONS / "global_shell_assembly.py").read_text(encoding="utf-8")
    assert "mitc4_stiffness(" not in src
    assert "assemble_section_from_zones(" in src
    thin = (_OPERATIONS / "assembly" / "section_assembly.py").read_text(encoding="utf-8")
    assert "build_formulation_cache(" in thin
    assert "assemble_section_system(" in thin
    # R8: scatter-add implementation lives in operations.assembly
    ops = (_OPERATIONS / "assembly" / "assemble_global.py").read_text(encoding="utf-8")
    assert "def assemble_section_system(" in ops
    assert "from ...engine.section_assembly import assemble_section_system" not in ops
    # Legacy root paths remain as re-export shims
    assert (_OPERATIONS / "section_assembly.py").is_file()
    assert (_OPERATIONS / "assemble_global.py").is_file()


def test_connected_assembly_core_has_no_legacy_solve_adapter():
    """``reduce_propagate_solve`` thin adapter was removed (Wave 3); solve via runner."""
    src = (_OPERATIONS / "connected_assembly_core.py").read_text(encoding="utf-8")
    assert "reduce_propagate_solve" not in src
    assert "SectionLinearStaticRunner" not in src
    runner_src = (_PKG / "simulation_runner" / "section_linear_static_runner.py").read_text(encoding="utf-8")
    assert "class SectionLinearStaticRunner" in runner_src


def test_engine_package_removed():
    """``engine/`` relocated into pre_processing / processing (Wave 6)."""
    assert not (_PKG / "engine").exists()
    assert _OPERATIONS.is_dir()
    assert (_PKG / "pre_processing" / "dispatch" / "shell_theory_registry.py").is_file()
    assert (_PKG / "pre_processing" / "loading" / "thin_wall" / "section_vlasov.py").is_file()
    assert not (_PKG / "pre_processing" / "loading" / "engine").exists()
    assert (_PKG / "pre_processing" / "loading" / "builders").is_dir()
    assert (_PKG / "pre_processing" / "loading" / "traction_face_load.py").is_file()
    assert not (_PKG / "interface" / "traction_face_load.py").exists()


def test_simulation_runner_owns_orchestration():
    src = (_PKG / "simulation_runner" / "section_linear_static_runner.py").read_text(encoding="utf-8")
    assert "def prepare_local_system" in src
    assert "def solve_linear_system_only" in src
    assert "def compute_primary_results" in src


def test_generic_result_orchestrators_have_no_hardcoded_formulation_id():
    for rel in (
        "processing/static/results/compute_tertiary/tertiary_results_orchestrator.py",
        "processing/static/results/compute_quaternary/quaternary_results_orchestrator.py",
    ):
        src = (_PKG / rel).read_text(encoding="utf-8")
        assert "MITC4-FSDT" not in src, rel
