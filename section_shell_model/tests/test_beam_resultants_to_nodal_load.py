"""Phase 0: R→f nodal equivalence and linear-static full-cascade smoke."""

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
    integrate_nodal_load_resultants,
)
from blade_structure_utilities.section_shell_model.pre_processing.mesh.api.pipeline_b import (
    build_pipeline_b_section,
)
from blade_structure_utilities.section_shell_model.processing.static.operations.section_assembly import (
    assemble_section_from_panels,
)
from blade_structure_utilities.section_shell_model.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.section_shell_model.simulation_runner.api.linear_static import (
    governing_fi_from_quaternary,
    run_section_linear_static_from_resultants,
)


def _asm_naca():
    pb = build_pipeline_b_section("2412", [0.35], chord_m=1.0)
    panels = pb.panels
    nx = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels]
    nxy = [np.zeros_like(a) for a in nx]
    asm = assemble_section_from_panels(
        panels, nx, nxy, n_elements_per_panel=4, shell_theory=ShellTheory.FSDT,
    )
    return pb, asm


@pytest.mark.parametrize(
    "named",
    [
        dict(N=1e3),
        dict(Vy=500.0),
        dict(Vz=500.0),
        dict(My=200.0),
        dict(Mz=200.0),
        dict(T=50.0),
        dict(N=1e3, Vy=200.0, Vz=-150.0, My=80.0, Mz=-40.0, T=25.0),
    ],
)
def test_nodal_load_reconstructs_beam_resultants(named):
    _pb, asm = _asm_naca()
    vec = BeamResultantVector.from_named(
        **{k: float(v) for k, v in named.items()},
        frame=FRAME_SECTION_YZ,
        about_yz=BEAM_LOAD_AXIS_YZ,
    )
    ndof = int(asm.cache.n_dof_node)
    f = beam_resultants_to_nodal_load(asm.node_meta, vec, ndof=ndof)
    back = integrate_nodal_load_resultants(
        asm.node_meta, f, about_yz=BEAM_LOAD_AXIS_YZ, ndof=ndof,
    )
    np.testing.assert_allclose(back.R, vec.R, rtol=1e-6, atol=1e-6)


def test_linear_static_cascade_reaches_quaternary():
    pb = build_pipeline_b_section("2412", [0.35], chord_m=1.0)
    vec = BeamResultantVector.from_named(
        My=1e3, frame=FRAME_SECTION_YZ, about_yz=BEAM_LOAD_AXIS_YZ,
    )
    out = run_section_linear_static_from_resultants(
        pb,
        resultants=vec,
        n_elements_per_panel=4,
        compute_section_deformation_1a=True,
    )
    assert out.quaternary is not None
    assert out.runner.primary_results is not None
    assert out.runner.secondary_results is not None
    assert out.runner.tertiary_results is not None
    fi, _nr = governing_fi_from_quaternary(out.quaternary)
    assert np.isfinite(fi)
    diag = out.runner.primary_results.global_results.diagnostics
    assert "section_deformation_1a" in diag
    avg = diag["section_deformation_1a"]
    assert avg.n_pairs > 0
    assert avg.mean_local_dof.shape == (6,)


def test_linear_static_from_resultants_rejects_non_fsdt() -> None:
    pb = build_pipeline_b_section("2412", [0.35], chord_m=1.0)
    with pytest.raises(ValueError, match="FSDT"):
        run_section_linear_static_from_resultants(
            pb,
            My=1.0e3,
            n_elements_per_panel=3,
            min_elements_per_panel=2,
            shell_theory=ShellTheory.CALCULIX_CCX,
        )
