"""Linear-static face-traction RHS: conservation, combined load, cascade FI."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.section_contract.beam_resultant_vector import (
    BEAM_LOAD_AXIS_YZ,
    FRAME_SECTION_YZ,
    BeamResultantVector,
)
from blade_structure_utilities.section_shell_model.pre_processing.loading.beam_resultants_to_nodal_load import (
    beam_resultants_to_nodal_load,
)
from blade_structure_utilities.section_shell_model.pre_processing.loading.traction_face_load import (
    applied_force_resultant_yz,
    traction_face_loads,
)
from blade_structure_utilities.section_shell_model.pre_processing.mesh.api.pipeline_b import (
    build_pipeline_b_section,
)
from blade_structure_utilities.section_shell_model.processing.static.operations.section_assembly import (
    assemble_section_from_panels,
)
from blade_structure_utilities.section_shell_model.simulation_runner.api.linear_static import (
    governing_fi_from_quaternary,
    run_section_linear_static_from_resultants,
)
from blade_structure_utilities.point_to_surftraction.tractions import TractionField


def _pb():
    return build_pipeline_b_section("2412", [0.35], chord_m=1.0)


def _assemble(pb, *, ne: int = 4):
    panels = list(pb.panels)
    nx = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels]
    nxy = [np.zeros_like(a) for a in nx]
    return assemble_section_from_panels(
        panels, nx, nxy, L_x=1.0, n_elements_per_panel=ne, min_elements_per_panel=2,
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
    pb = _pb()
    asm = _assemble(pb)
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
    pb = _pb()
    tf, pts = _synthetic_traction()
    with pytest.raises(ValueError, match="must be supplied together"):
        run_section_linear_static_from_resultants(
            pb, My=1.0e3, traction_field=tf, n_elements_per_panel=3,
        )
    with pytest.raises(ValueError, match="must be supplied together"):
        run_section_linear_static_from_resultants(
            pb, My=1.0e3, traction_points_yz=pts, n_elements_per_panel=3,
        )


def test_combined_rhs_equals_beam_plus_traction() -> None:
    pb = _pb()
    asm = _assemble(pb, ne=3)
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
    cascade = run_section_linear_static_from_resultants(
        pb,
        resultants=resultants,
        traction_field=tf,
        traction_points_yz=pts,
        n_elements_per_panel=3,
        min_elements_per_panel=2,
    )
    assert cascade.used_traction is True
    np.testing.assert_allclose(cascade.f_nodal, f_beam + f_tr, rtol=0.0, atol=1e-12)


def test_linear_static_traction_cascade_reaches_quaternary_fi() -> None:
    pb = _pb()
    tf, pts = _synthetic_traction(q_y=5.0e3, q_z=1.0e3)
    cascade = run_section_linear_static_from_resultants(
        pb,
        T=10.0,
        traction_field=tf,
        traction_points_yz=pts,
        n_elements_per_panel=3,
        min_elements_per_panel=2,
    )
    fi, _needs = governing_fi_from_quaternary(cascade.quaternary)
    assert np.isfinite(fi)
    assert cascade.used_traction is True
    assert cascade.quaternary is not None
