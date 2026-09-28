"""Architecture gates for the FE-row ownership matrix."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = "blade_structure_utilities.fem_utilities.section_shell_fea"

BANNED_TOP_LEVEL_ABSENT = ("engine", "core")
NAMED_EXCEPTION_DIRS = ("interface", "pre_processing", "processing", "simulation_runner")

# Policy / shaping symbols forbidden in FE-SOLVE / FE-CASCADE production homes.
_FORBIDDEN_BIND_SYMBOLS = frozenset(
    {
        "bind_shell_material_zones",
        "shell_material_zones_from_mesh_inputs",
        "SectionBindRunner",
    }
)
_FORBIDDEN_LOADS_SYMBOLS = frozenset(
    {
        "shape_section_force_vector",
        "shape_section_force_vector_from_panel_topology",
        "SectionLoadsRunner",
        "distribute_loads",
        "distribute_beam_resultants",
        "equipollent_shape_on_midlines",
    }
)

BARREL_ALLOWLIST = frozenset(
    {
        "SectionLinearStaticRunner",
        "SectionLinearStaticResult",
        "CascadeBin",
        "DiagnosticsBin",
        "LinearBucklingCascadeResult",
        "run_section_linear_buckling_from_static",
        "BoundSectionShellInputs",
        "extract_shell_full_element_results",
    }
)


def test_banned_software_layer_homes_absent() -> None:
    for name in BANNED_TOP_LEVEL_ABSENT:
        assert not (ROOT / name).exists(), f"banned home still present: {name}/"


def test_named_staging_exceptions_exist() -> None:
    for name in NAMED_EXCEPTION_DIRS:
        assert (ROOT / name).is_dir(), f"named FE staging home missing: {name}/"
    assert (ROOT / "pre_processing" / "containers").is_dir()
    assert (ROOT / "calculix_ccx").is_dir()


def test_fe_row_docs_exist() -> None:
    assert (ROOT / "docs" / "reference" / "FE_ROW_OWNERSHIP_MATRIX.md").is_file()
    assert (ROOT / "simulation_runner").is_dir()
    assert (ROOT / "pre_processing").is_dir()
    assert (ROOT / "processing" / "static").is_dir()
    assert (ROOT / "processing" / "buckling").is_dir()


def _iter_py_files(rel: str) -> list[Path]:
    root = ROOT / rel
    if not root.is_dir():
        return []
    return [
        p
        for p in root.rglob("*.py")
        if p.is_file() and "tests" not in p.relative_to(ROOT).parts
    ]


def _imported_names(path: Path) -> set[tuple[str, str]]:
    """Return (module, name) pairs from import statements."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(ROOT)
    file_pkg_parts = [PKG] + list(rel.with_suffix("").parts)
    parent_parts = file_pkg_parts[:-1]
    out: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add((alias.name, alias.asname or alias.name.split(".")[-1]))
        elif isinstance(node, ast.ImportFrom):
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


@pytest.mark.parametrize(
    "home",
    [
        "processing/static/operations",
        "processing/static/results",
        "processing/static/constraints",
    ],
)
def test_solve_cascade_forbid_bind_policy_and_sfv_shaping(home: str) -> None:
    violations: list[str] = []
    for path in _iter_py_files(home):
        for mod, name in _imported_names(path):
            if name in _FORBIDDEN_BIND_SYMBOLS or (
                "section_bind" in mod and name in _FORBIDDEN_BIND_SYMBOLS
            ):
                violations.append(
                    f"{path.relative_to(ROOT).as_posix()} imports {mod}.{name}"
                )
            if name in _FORBIDDEN_LOADS_SYMBOLS or (
                "section_loads" in mod and name in _FORBIDDEN_LOADS_SYMBOLS
            ):
                violations.append(
                    f"{path.relative_to(ROOT).as_posix()} imports {mod}.{name}"
                )
            if mod == PKG:
                violations.append(
                    f"{path.relative_to(ROOT).as_posix()} imports package barrel"
                )
    assert not violations, "FE-SOLVE/CASCADE skip-layer imports:\n" + "\n".join(
        violations
    )


def test_production_runner_docstring_uses_frozen_sfv() -> None:
    path = ROOT / "simulation_runner" / "section_linear_static_runner.py"
    text = path.read_text(encoding="utf-8")
    doc_start = text.find('"""Classical FEM facade')
    assert doc_start >= 0
    doc_end = text.find('"""', doc_start + 3)
    docstring = text[doc_start:doc_end]
    assert "shape_section_force_vector" not in docstring
    assert "section_force_vector" in docstring


def test_station_recovery_does_not_shape_sfv() -> None:
    path = ROOT / "simulation_runner" / "api" / "station_recovery.py"
    # No import-time / runtime call of the shaping helpers (docstring may name them).
    imports = _imported_names(path)
    for mod, name in imports:
        assert name not in _FORBIDDEN_LOADS_SYMBOLS, f"station_recovery imports {mod}.{name}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            label = ""
            if isinstance(func, ast.Name):
                label = func.id
            elif isinstance(func, ast.Attribute):
                label = func.attr
            assert label not in _FORBIDDEN_LOADS_SYMBOLS, (
                f"station_recovery calls {label}"
            )


def test_barrel_allowlist() -> None:
    import blade_structure_utilities.fem_utilities.section_shell_fea as fea

    exported = set(getattr(fea, "__all__", ()))
    unexpected = exported - BARREL_ALLOWLIST
    missing = BARREL_ALLOWLIST - exported
    assert not unexpected, f"barrel exports undeclared symbols: {sorted(unexpected)}"
    assert not missing, f"barrel missing documented symbols: {sorted(missing)}"
    assert fea.BoundSectionShellInputs is not None
    assert fea.extract_shell_full_element_results is not None
    assert fea.SectionLinearStaticRunner is not None