"""WP5: FE loading adapter import contracts and runner refusal."""

from __future__ import annotations

from pathlib import Path

_FE_ROOT = Path(__file__).resolve().parents[1]
_LOADING = _FE_ROOT / "pre_processing" / "loading"


def _iter_prod_py(root: Path) -> list[Path]:
    skip = {"tests", "visualisation", "examples", "docs", "archive", "__pycache__"}
    out: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in skip for part in path.parts):
            continue
        out.append(path)
    return out


def test_loading_exports_fe_adapters_only() -> None:
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing import (
        loading as loading_pkg,
    )

    banned = {
        "DistributedStationLoadSpec",
        "IncidentAeroSpec",
        "LoadFieldProvenance",
        "SectionLoadsJobConfig",
        "SectionStationLoadField",
        "StationAeroContext",
        "build_station_load_field",
        "SectionLoadsRunner",
        "shape_section_force_vector",
    }
    exported = set(loading_pkg.__all__)
    assert not (exported & banned), f"loading barrel re-exports BSU loads: {exported & banned}"
    assert "shear_centre_about_yz_from_panel_topology" in exported
    assert "beam_resultants_to_nodal_load" in exported


def test_no_section_loads_runner_execute_in_fe_facades() -> None:
    hits: list[str] = []
    for path in _iter_prod_py(_FE_ROOT / "pre_processing"):
        text = path.read_text(encoding="utf-8")
        if "SectionLoadsRunner" in text and ".execute(" in text:
            hits.append(path.relative_to(_FE_ROOT).as_posix())
    assert not hits, "FE facade must not execute SectionLoadsRunner:\n" + "\n".join(hits)


def test_shear_centre_adapter_does_not_import_cap_tie_geometry() -> None:
    text = (_LOADING / "shear_centre_loading_axis.py").read_text(encoding="utf-8")
    assert "cap_tie.geometry" not in text
    assert "processing.static.constraints" not in text


def test_beam_resultants_module_is_quarantined() -> None:
    text = (_LOADING / "beam_resultants_to_nodal_load.py").read_text(encoding="utf-8")
    assert "Regression-only" in text or "regression" in text.lower()
    assert "Not a production apply path" in text or "not a production" in text.lower()


def test_no_load_shaping_algorithm_under_fe_adapters() -> None:
    """FE loading must not host Channel A/B shaping kernels."""
    banned_snippets = (
        "def shape_section_force_vector",
        "class SectionLoadsRunner",
        "def build_station_load_field",
    )
    hits: list[str] = []
    for path in _iter_prod_py(_LOADING):
        # Shim re-exports under builders/api/containers are allowed until expiry.
        rel = path.relative_to(_LOADING).as_posix()
        if rel.startswith(("builders/", "api/", "containers/", "io/")):
            continue
        text = path.read_text(encoding="utf-8")
        for snip in banned_snippets:
            if snip in text:
                hits.append(f"{rel}: {snip}")
    assert not hits, "load-shaping algorithm duplicated under FE:\n" + "\n".join(hits)
