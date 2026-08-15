"""Linear-static face-traction RHS: conservation, combined load, cascade FI."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BEAM_LOAD_AXIS_YZ,
    FRAME_SECTION_YZ,
    BeamResultantVector,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.beam_resultants_to_nodal_load import (
    beam_resultants_to_nodal_load,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.traction_face_load import (
    applied_force_resultant_yz,
    traction_face_loads,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    build_shell_panel_topology,
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
from blade_structure_utilities.load_utilities.section_loads import TractionField
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


def _synthetic_traction(n_panels: int = 4, q_y: float = 100.0, q_z: float = 20.0):
    forces = np.zeros((n_panels, 2), dtype=float)
    forces[:, 0] = q_y / n_panels
    forces[:, 1] = q_z / n_panels
    # Place panel centres along a chord-ish line covering the section.
    pts = np.column_stack(
        [
            np.linspace(-0.2, 0.6, n_panels),
            np.zeros(n_panels),
        ]
    )
    tf = TractionField(
        pressure_forces=forces.copy(),
        shear_forces=np.zeros_like(forces),
        total_forces=forces,
    )
    return tf, pts


def test_traction_face_loads_conserves_total_force() -> None:
    panel_topology = _pb()
    asm = _assemble(panel_topology)
    ndof = int(asm.cache.n_dof_node)
    tf, pts = _synthetic_traction()
    L_x = 1.0
    f_tr = traction_face_loads(
        asm.node_meta,
        np.asarray(tf.total_forces, dtype=float),
        pts,
        ndof=ndof,
        L_x=L_x,
    )
    recovered = applied_force_resultant_yz(asm.node_meta, f_tr, ndof=ndof)
    expected = L_x * np.sum(np.asarray(tf.total_forces, dtype=float), axis=0)
    np.testing.assert_allclose(recovered, expected, rtol=0.0, atol=1e-9)


def test_half_specified_traction_raises() -> None:
    panel_topology = _pb()
    mesh = _strip_mesh(panel_topology)
    tf, pts = _synthetic_traction()
    with pytest.raises(ValueError, match="must be supplied together"):
        LinearStaticCascadeResult.from_executed_runner(
            SectionLinearStaticRunner(
                panel_topology=panel_topology,
                My=1.0e3,
                traction_field=tf,
                section_strip_mesh=mesh,
                load_channel="equipollent",
            ).execute()
        )
    with pytest.raises(ValueError, match="must be supplied together"):
        LinearStaticCascadeResult.from_executed_runner(
            SectionLinearStaticRunner(
                panel_topology=panel_topology,
                My=1.0e3,
                traction_points_yz=pts,
                section_strip_mesh=mesh,
                load_channel="equipollent",
            ).execute()
        )


def test_combined_rhs_equals_beam_plus_traction() -> None:
    panel_topology = _pb()
    asm = _assemble(panel_topology, ne=3)
    ndof = int(asm.cache.n_dof_node)
    resultants = BeamResultantVector.from_named(
        N=0.0, Vy=0.0, Vz=0.0, My=1.0e3, Mz=0.0, T=12.0,
        frame=FRAME_SECTION_YZ,
        about_yz=BEAM_LOAD_AXIS_YZ,
    )
    tf, pts = _synthetic_traction()
    f_beam = beam_resultants_to_nodal_load(
        asm.node_meta, resultants, about_yz=resultants.about_yz, ndof=ndof, L_x=1.0,
    )
    f_tr = traction_face_loads(
        asm.node_meta,
        np.asarray(tf.total_forces, dtype=float),
        pts,
        ndof=ndof,
        L_x=1.0,
    )
    cascade = LinearStaticCascadeResult.from_executed_runner(
        SectionLinearStaticRunner(
            panel_topology=panel_topology,
            resultants=resultants,
            traction_field=tf,
            traction_points_yz=pts,
            section_strip_mesh=_strip_mesh(panel_topology),
            load_channel="equipollent",
        ).execute()
    )
    assert cascade.used_traction is True
    np.testing.assert_allclose(cascade.f_nodal, f_beam + f_tr, rtol=0.0, atol=1e-12)


def test_linear_static_traction_cascade_reaches_quaternary_fi() -> None:
    panel_topology = _pb()
    tf, pts = _synthetic_traction(q_y=5.0e3, q_z=1.0e3)
    cascade = LinearStaticCascadeResult.from_executed_runner(
        SectionLinearStaticRunner(
            panel_topology=panel_topology,
            T=10.0,
            traction_field=tf,
            traction_points_yz=pts,
            section_strip_mesh=_strip_mesh(panel_topology),
            load_channel="equipollent",
        ).execute()
    )
    fi, _needs = governing_fi_from_quaternary(cascade.quaternary)
    assert np.isfinite(fi)
    assert cascade.used_traction is True
    assert cascade.quaternary is not None
