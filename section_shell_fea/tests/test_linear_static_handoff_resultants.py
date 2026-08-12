"""Linear-static cascade handoff exposes nested secondary resultants."""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    build_shell_panel_topology,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.api.linear_static import (
    LinearStaticCascadeResult,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.api.station_recovery import (
    _bundle_from_linear_cascade,
    build_fsdt_station_analysis,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)
from blade_structure_utilities.mesh_utilities.section_mesh.engine.strip_mesh import (
    build_section_strip_mesh,
)


def _pb():
    return build_shell_panel_topology("2412", [0.35], chord_m=1.0)


def _strip_mesh(panel_topology, *, n_elements_per_panel: int = 3):
    return build_section_strip_mesh(
        panel_topology.shell_inputs,
        n_elements_per_panel=n_elements_per_panel,
        min_elements_per_panel=2,
    )


def test_linear_handoff_all_panel_results_nonempty_for_my() -> None:
    panel_topology = _pb()
    mesh = _strip_mesh(panel_topology)
    cascade = LinearStaticCascadeResult.from_executed_runner(
        SectionLinearStaticRunner(
            panel_topology=panel_topology,
            My=1.0e3,
            section_strip_mesh=mesh,
            load_channel="equipollent",
        ).execute()
    )
    bundle = _bundle_from_linear_cascade(panel_topology, cascade)
    nested = list(bundle.all_panel_results or [])
    assert nested, "expected nested panel rows"
    assert any(len(panel_rows or []) > 0 for panel_rows in nested)
    assert bundle.reference_resultants is not None


def test_station_analysis_sees_cascade_shell_resultants() -> None:
    panel_topology = _pb()
    analysis = build_fsdt_station_analysis(
        panel_topology.section,
        panel_topology.shell_inputs,
        z_station=0.35,
        My=1.0e3,
        section_strip_mesh=_strip_mesh(panel_topology),
        station_id="handoff-test",
    )
    assert len(analysis.shell_resultants) > 0
