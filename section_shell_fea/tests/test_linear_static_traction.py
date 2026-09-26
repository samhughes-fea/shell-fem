"""FEA-L-003: refuse proximity traction; Channel B via labelled SFV only."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BEAM_LOAD_AXIS_YZ,
    FRAME_SECTION_YZ,
    BeamResultantVector,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.traction_face_load import (
    traction_face_loads,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.tests._panel_topology_factory import (
    build_shell_panel_topology,
    equipollent_section_force_vector,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.section_assembly import (
    assemble_section_from_zones,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.api.linear_static import (
    LinearStaticCascadeResult,
    governing_fi_from_quaternary,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)
from blade_structure_utilities.load_utilities.section_loads import (
    TractionField,
    traction_field_to_section_force_vector,
)
from blade_structure_utilities.load_utilities.section_loads.surface_traction.geometry import (
    SectionOutlineGeometry,
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


def _assemble(panel_topology, *, ne: int = 4):
    zones = list(panel_topology.zones)
    nx = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in zones]
    nxy = [np.zeros_like(a) for a in nx]
    return assemble_section_from_zones(
        zones,
        nx,
        nxy,
        L_x=1.0,
        section_strip_mesh=_strip_mesh(panel_topology, n_elements_per_panel=ne),
    )


def test_traction_face_loads_refused() -> None:
    panel_topology = _pb()
    asm = _assemble(panel_topology)
    ndof = int(asm.cache.n_dof_node)
    with pytest.raises(ValueError, match="FEA-L-003"):
        traction_face_loads(
            asm.node_meta,
            np.ones((2, 2)),
            np.zeros((2, 2)),
            ndof=ndof,
            L_x=1.0,
        )


def test_runner_rejects_traction_overlay_kwargs() -> None:
    panel_topology = _pb()
    mesh = _strip_mesh(panel_topology)
    resultants = BeamResultantVector.from_named(
        My=1.0e3, frame=FRAME_SECTION_YZ, about_xy=BEAM_LOAD_AXIS_YZ,
    )
    sfv = equipollent_section_force_vector(panel_topology, resultants)
    with pytest.raises(TypeError, match="traction_field"):
        SectionLinearStaticRunner(
            panel_topology=panel_topology,
            My=1.0e3,
            section_force_vector=sfv,
            traction_field=object(),
            section_strip_mesh=mesh,
            load_channel="equipollent",
        )


def test_channel_b_sfv_applies_via_fea_l001() -> None:
    """Channel B: labelled SFV → consistent nodal loads (no proximity)."""
    panel_topology = _pb()
    midlines = list(panel_topology.shell_inputs.midlines)
    outline = SectionOutlineGeometry.from_skin_midlines(midlines)
    n = outline.n_panels
    forces = np.zeros((n, 2), dtype=float)
    forces[:, 0] = 100.0 / n
    forces[:, 1] = 20.0 / n
    tf = TractionField(
        pressure_forces=forces.copy(),
        shear_forces=np.zeros_like(forces),
        total_forces=forces.copy(),
    )
    sfv = traction_field_to_section_force_vector(tf, outline=outline, L_x=1.0)
    asm = _assemble(panel_topology, ne=3)
    ndof = int(asm.cache.n_dof_node)
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.section_force_vector_to_nodal_load import (
        section_force_vector_to_nodal_load,
    )

    f_sfv = section_force_vector_to_nodal_load(
        asm.node_meta,
        sfv,
        ndof=ndof,
        L_x=1.0,
        panel_maps=asm.panel_maps,
        zones=panel_topology.zones,
    )
    cascade = LinearStaticCascadeResult.from_executed_runner(
        SectionLinearStaticRunner(
            panel_topology=panel_topology,
            section_force_vector=sfv,
            section_strip_mesh=_strip_mesh(panel_topology),
            load_channel="surface_traction",
        ).execute()
    )
    np.testing.assert_allclose(cascade.f_nodal, f_sfv, rtol=0.0, atol=1e-12)
    assert cascade.runner.load_channel_used == "surface_traction"


def test_equipollent_cascade_reaches_quaternary_fi() -> None:
    panel_topology = _pb()
    resultants = BeamResultantVector.from_named(
        T=10.0, frame=FRAME_SECTION_YZ, about_xy=BEAM_LOAD_AXIS_YZ,
    )
    sfv = equipollent_section_force_vector(panel_topology, resultants)
    cascade = LinearStaticCascadeResult.from_executed_runner(
        SectionLinearStaticRunner(
            panel_topology=panel_topology,
            resultants=resultants,
            section_force_vector=sfv,
            section_strip_mesh=_strip_mesh(panel_topology),
            load_channel="equipollent",
        ).execute()
    )
    fi, _needs = governing_fi_from_quaternary(cascade.quaternary)
    assert np.isfinite(fi)
    assert cascade.quaternary is not None
