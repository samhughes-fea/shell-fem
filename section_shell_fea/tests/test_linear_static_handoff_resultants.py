"""Linear-static cascade handoff exposes nested secondary resultants."""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea.tests._panel_topology_factory import (
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
from blade_structure_utilities.mesh_utilities.section_mesh.strip.strip_mesh import (
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
    from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
        BEAM_LOAD_AXIS_YZ,
        FRAME_SECTION_YZ,
        BeamResultantVector,
    )
    from blade_structure_utilities.load_utilities.section_loads import (
        shape_section_force_vector_from_panel_topology,
    )

    panel_topology = _pb()
    resultants = BeamResultantVector.from_named(
        My=1.0e3,
        frame=FRAME_SECTION_YZ,
        about_xy=BEAM_LOAD_AXIS_YZ,
    )
    sfv = shape_section_force_vector_from_panel_topology(resultants, panel_topology)
    analysis = build_fsdt_station_analysis(
        panel_topology.section,
        panel_topology.shell_inputs,
        z_station=0.35,
        My=1.0e3,
        section_strip_mesh=_strip_mesh(panel_topology),
        section_force_vector=sfv,
        panel_topology=panel_topology,
        station_id="handoff-test",
    )
    assert len(analysis.shell_resultants) > 0

