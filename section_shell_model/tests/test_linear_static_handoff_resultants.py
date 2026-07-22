"""Linear-static cascade handoff exposes nested secondary resultants."""

from __future__ import annotations

from blade_structure_utilities.section_shell_model.pre_processing.mesh.api.pipeline_b import (
    build_pipeline_b_section,
)
from blade_structure_utilities.section_shell_model.simulation_runner.api.linear_static import (
    run_section_linear_static_from_resultants,
)
from blade_structure_utilities.section_shell_model.simulation_runner.api.station_recovery import (
    _bundle_from_linear_cascade,
    build_fsdt_station_analysis,
)


def _pb():
    return build_pipeline_b_section("2412", [0.35], chord_m=1.0)


def test_linear_handoff_all_panel_results_nonempty_for_my() -> None:
    pb = _pb()
    cascade = run_section_linear_static_from_resultants(
        pb,
        My=1.0e3,
        n_elements_per_panel=3,
        min_elements_per_panel=2,
    )
    bundle = _bundle_from_linear_cascade(pb, cascade)
    nested = list(bundle.all_panel_results or [])
    assert nested, "expected nested panel rows"
    assert any(len(panel_rows or []) > 0 for panel_rows in nested)
    assert bundle.reference_resultants is not None


def test_station_analysis_sees_cascade_shell_resultants() -> None:
    pb = _pb()
    analysis = build_fsdt_station_analysis(
        pb.section,
        pb.shell_inputs,
        z_station=0.35,
        My=1.0e3,
        n_elements_per_panel=3,
        min_elements_per_panel=2,
        station_id="handoff-test",
    )
    assert len(analysis.shell_resultants) > 0
