"""Static boundary: legacy ``lib.laminate_clpt`` / stress-model shims stay contained."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def _iter_production_py_files() -> list[Path]:
    root = _REPO / "blade_precompute"
    skip_parts = {"tests", "examples", "benchmarks", "__pycache__"}
    out: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in skip_parts for part in path.parts):
            continue
        out.append(path)
    return out


def _rel(path: Path) -> str:
    return path.relative_to(_REPO).as_posix()


# Frozen allow-lists: new production callers must not be added without review.
_LIB_LAMINATE_CLPT_ALLOW: set[str] = set()

# Production ``ensure_stress_model_lib_path`` callers (section_shell_model cleared).
_ENSURE_STRESS_IMPORTS_ALLOW = {
    "blade_precompute/_vendor_paths.py",
}

# Production ``multi_cell_blade_section`` string hits (imports or docs). Empty after R12.
_MULTI_CELL_ALLOW: set[str] = set()

_SYS_PATH_STRESS_MODEL_ALLOW = {
    "blade_precompute/_vendor_paths.py",
}


def _has_lib_laminate_clpt_import(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "from lib.laminate_clpt" in stripped or "import lib.laminate_clpt" in stripped:
            return True
    return False


def test_section_shell_model_production_has_no_lib_laminate_clpt_imports() -> None:
    pkg = _REPO / "blade_precompute" / "section_shell_model"
    offenders: list[str] = []
    for path in pkg.rglob("*.py"):
        if "tests" in path.parts or "examples" in path.parts or "benchmarks" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if _has_lib_laminate_clpt_import(text):
            offenders.append(_rel(path))
    assert offenders == [], f"Unexpected lib.laminate_clpt imports: {offenders}"


def test_blade_precompute_lib_laminate_clpt_imports_are_allowlisted() -> None:
    offenders: list[str] = []
    for path in _iter_production_py_files():
        text = path.read_text(encoding="utf-8")
        if not _has_lib_laminate_clpt_import(text):
            continue
        rel = _rel(path)
        if rel not in _LIB_LAMINATE_CLPT_ALLOW:
            offenders.append(rel)
    assert offenders == [], f"New lib.laminate_clpt production imports: {offenders}"


def test_ensure_stress_imports_callers_are_allowlisted() -> None:
    offenders: list[str] = []
    for path in _iter_production_py_files():
        text = path.read_text(encoding="utf-8")
        if "_ensure_stress_imports" not in text and "ensure_stress_model_lib_path" not in text:
            continue
        rel = _rel(path)
        if rel not in _ENSURE_STRESS_IMPORTS_ALLOW:
            offenders.append(rel)
    assert offenders == [], f"New _ensure_stress_imports callers: {offenders}"


def test_multi_cell_blade_section_imports_are_allowlisted() -> None:
    offenders: list[str] = []
    for path in _iter_production_py_files():
        text = path.read_text(encoding="utf-8")
        if "multi_cell_blade_section" not in text:
            continue
        rel = _rel(path)
        if rel not in _MULTI_CELL_ALLOW:
            offenders.append(rel)
    assert offenders == [], f"New multi_cell_blade_section production imports: {offenders}"


def test_direct_section_stress_model_sys_path_shims_are_allowlisted() -> None:
    offenders: list[str] = []
    needle = "section_stress_model"
    for path in _iter_production_py_files():
        text = path.read_text(encoding="utf-8")
        if needle not in text or "sys.path" not in text:
            continue
        rel = _rel(path)
        if rel in _SYS_PATH_STRESS_MODEL_ALLOW:
            continue
        if rel == "blade_precompute/_vendor_paths.py":
            continue
        if "sys.path.insert" in text and needle in text:
            offenders.append(rel)
    assert offenders == [], f"New section_stress_model sys.path shims: {offenders}"


def test_topology_has_no_vendor_bootstrap() -> None:
    """R12 complete: topology defaults are package-local (no vendor sys.path)."""
    path = (
        _REPO
        / "blade_precompute"
        / "section_shell_model"
        / "pre_processing"
        / "mesh"
        / "topology.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "multi_cell_blade_section" not in text
    assert "ensure_stress_model_lib_path" not in text
    assert "default_skin_web_cap_laminates" in text
    assert "_default_laminates" in text
