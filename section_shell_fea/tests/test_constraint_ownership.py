"""Ownership gates for the constraints / assembly / SFV architecture."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "blade_structure_utilities.fem_utilities.section_shell_fea"


def _iter_py(rel: str):
    root = ROOT / rel
    if not root.is_dir():
        return []
    return [
        p
        for p in root.rglob("*.py")
        if p.is_file() and "tests" not in p.relative_to(ROOT).parts
    ]


def _imported_names(path: Path) -> set[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(ROOT)
    file_pkg_parts = [PKG] + list(rel.with_suffix("").parts)
    parent_parts = file_pkg_parts[:-1]
    out: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                mod = node.module or ""
            else:
                base = list(parent_parts)
                for _ in range(node.level):
                    if base:
                        base.pop()
                if node.module:
                    mod = ".".join(base + node.module.split("."))
                else:
                    mod = ".".join(base)
            for alias in node.names:
                out.add((mod, alias.name))
    return out


def test_constraints_home_exists():
    assert (ROOT / "processing" / "static" / "constraints").is_dir()
    assert (ROOT / "processing" / "static" / "constraints" / "compose.py").is_file()
    assert (ROOT / "pre_processing" / "containers" / "analysis_dof_layout.py").is_file()
    assert (ROOT / "pre_processing" / "mesh" / "global_numbering.py").is_file()


def test_mesh_prep_does_not_import_processing_solve():
    violations = []
    for path in _iter_py("pre_processing/mesh"):
        for mod, name in _imported_names(path):
            if ".processing." in mod or mod.endswith(".processing"):
                violations.append(f"{path.relative_to(ROOT).as_posix()} imports {mod}.{name}")
    assert not violations, "pre_processing.mesh imported processing:\n" + "\n".join(violations)


def test_constraints_do_not_shape_sfv_or_bind():
    banned = {
        "shape_section_force_vector",
        "SectionLoadsRunner",
        "bind_shell_material_zones",
        "SectionBindRunner",
        "equipollent_shape_on_midlines",
    }
    violations = []
    for path in _iter_py("processing/static/constraints"):
        for mod, name in _imported_names(path):
            if name in banned:
                violations.append(f"{path.relative_to(ROOT).as_posix()} imports {mod}.{name}")
    assert not violations, "constraints imported bind/load shaping:\n" + "\n".join(violations)


def test_production_runner_does_not_import_equipollent_direct_application():
    path = ROOT / "simulation_runner" / "section_linear_static_runner.py"
    banned = {
        "beam_resultants_to_nodal_load",
        "equipollent_shape_on_midlines",
        "shape_section_force_vector",
        "SectionLoadsRunner",
        "bind_shell_material_zones",
        "SectionBindRunner",
    }
    imported = {name for _mod, name in _imported_names(path)}
    assert not (imported & banned)
    text = path.read_text(encoding="utf-8")
    assert "equipollent_shape_on_midlines" in text
    assert "this runner only applies" in text
