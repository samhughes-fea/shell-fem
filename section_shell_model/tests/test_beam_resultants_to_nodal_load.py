"""Phase 0: R→f nodal equivalence and linear-static full-cascade smoke."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BEAM_LOAD_AXIS_YZ,
    FRAME_SECTION_YZ,
    BeamResultantVector,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.beam_resultants_to_nodal_load import (
    beam_resultants_to_nodal_load,
    integrate_nodal_load_resultants,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.api.shell_panel_topology import (
    build_shell_panel_topology,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.operations.section_assembly import (
    assemble_section_from_panels,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)


def _asm_naca():
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    panels = panel_topology.panels
    nx = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels]
    nxy = [np.zeros_like(a) for a in nx]
    asm = assemble_section_from_panels(
        panels, nx, nxy, n_elements_per_panel=4, shell_theory=ShellTheory.FSDT,
    )
    return panel_topology, asm


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
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    vec = BeamResultantVector.from_named(
        My=1e3, frame=FRAME_SECTION_YZ, about_yz=BEAM_LOAD_AXIS_YZ,
    )
    static = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        resultants=vec,
        n_elements_per_panel=4,
        load_channel="equipollent",
        compute_section_deformation_1a=True,
    ).execute()
    results = static.get_results()
    quaternary = results.quaternary_results
    assert quaternary is not None
    gov = quaternary.governing or {}
    fi = float(gov.get("fi", 0.0) or 0.0)
    assert np.isfinite(fi)
    assert static.primary_results is not None
    diag = static.primary_results.global_results.diagnostics
    assert "section_deformation_1a" in diag


def test_torsion_about_shear_centre_opt_in_reconstructs_T():
    """Pure T about sectorial SC reconstructs about that origin (opt-in path)."""
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.shear_centre_loading_axis import (
        shear_centre_about_yz_from_panel_topology,
    )

    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    y_sc, z_sc = shear_centre_about_yz_from_panel_topology(panel_topology)
    assert np.isfinite(y_sc) and np.isfinite(z_sc)

    static = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        T=50.0,
        torsion_about_shear_centre=True,
        n_elements_per_panel=4,
        load_channel="equipollent",
    ).execute()
    assert static.resultants is not None
    assert static.resultants.about_yz == pytest.approx((y_sc, z_sc), rel=0, abs=1e-9)
    assert static.assembly is not None and static.f_nodal is not None
    back = integrate_nodal_load_resultants(
        static.assembly.node_meta,
        static.f_nodal,
        about_yz=static.resultants.about_yz,
        ndof=int(static.assembly.cache.n_dof_node),
    )
    np.testing.assert_allclose(back.T, 50.0, rtol=1e-5, atol=1e-5)
    # Default product path unchanged when flag is off
    static_default = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        T=50.0,
        n_elements_per_panel=4,
        load_channel="equipollent",
    ).execute()
    assert static_default.resultants is not None
    assert static_default.resultants.about_yz == BEAM_LOAD_AXIS_YZ


def test_linear_static_from_resultants_rejects_non_fsdt() -> None:
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    with pytest.raises(ValueError, match="FSDT"):
        SectionLinearStaticRunner(
            panel_topology=panel_topology,
            My=1.0e3,
            n_elements_per_panel=3,
            min_elements_per_panel=2,
            load_channel="equipollent",
            shell_theory=ShellTheory.CALCULIX_CCX,
        ).execute()
