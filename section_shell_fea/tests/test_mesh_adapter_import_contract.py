"""WP4: FE mesh adapter import contracts and runner refusal."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_FE_ROOT = Path(__file__).resolve().parents[1]
_MESH = _FE_ROOT / "pre_processing" / "mesh"
_TOPOLOGY = _FE_ROOT / "pre_processing" / "topology"


def _iter_prod_py(root: Path) -> list[Path]:
    skip = {"tests", "visualisation", "examples", "docs", "archive", "__pycache__"}
    out: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in skip for part in path.parts):
            continue
        out.append(path)
    return out


def test_mesh_exports_fe_adapter_products_only() -> None:
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing import mesh as mesh_pkg

    banned = {
        "SectionMeshRunner",
        "build_section_strip_mesh",
        "shell_mesh_inputs_from_geometry_result",
    }
    exported = set(mesh_pkg.__all__)
    assert not (exported & banned), f"mesh barrel re-exports BSU mesh worker: {exported & banned}"
    assert "BoundSectionShellInputs" in exported
    assert "PanelNodeAssemblyMap" in exported
    assert (_MESH / "naca_profile.py").is_file()


def test_topology_package_is_shim_only() -> None:
    """topology/*.py must be logic-free warning aliases (no heavy algorithm bodies)."""
    for path in _iter_prod_py(_TOPOLOGY):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        assert not defs, f"topology shim still defines code: {path.relative_to(_FE_ROOT)} -> {[d.name for d in defs]}"


def test_no_section_mesh_runner_execute_in_fe_facades() -> None:
    hits: list[str] = []
    for path in _iter_prod_py(_FE_ROOT / "pre_processing"):
        text = path.read_text(encoding="utf-8")
        if "SectionMeshRunner" in text and ".execute(" in text:
            hits.append(path.relative_to(_FE_ROOT).as_posix())
        if "SectionMeshRunner().execute" in text or "SectionMeshRunner().execute()" in text:
            hits.append(path.relative_to(_FE_ROOT).as_posix())
    assert not hits, "FE facade must not execute SectionMeshRunner:\n" + "\n".join(hits)


def test_production_imports_prefer_mesh_home() -> None:
    """New production code under processing/simulation should import mesh, not topology."""
    hits: list[str] = []
    for path in _iter_prod_py(_FE_ROOT / "processing") + _iter_prod_py(_FE_ROOT / "simulation_runner"):
        text = path.read_text(encoding="utf-8")
        if "pre_processing.topology" in text:
            hits.append(path.relative_to(_FE_ROOT).as_posix())
    assert not hits, "processing/simulation still deep-import topology:\n" + "\n".join(hits)
