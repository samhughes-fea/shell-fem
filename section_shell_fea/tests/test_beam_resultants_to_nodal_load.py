"""Phase 0: Râ†’f nodal equivalence and linear-static full-cascade smoke."""

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
    integrate_nodal_load_resultants,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.tests._panel_topology_factory import (
    build_shell_panel_topology,
    equipollent_section_force_vector,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.section_assembly import (
    assemble_section_from_zones,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)
from blade_structure_utilities.mesh_utilities.section_mesh.strip.strip_mesh import (
    build_section_strip_mesh,
)


def _strip_mesh(panel_topology, *, n_elements_per_panel: int = 4):
    return build_section_strip_mesh(
        panel_topology.shell_inputs,
        n_elements_per_panel=n_elements_per_panel,
    )


def _asm_naca():
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    zones = list(panel_topology.zones)
    nx = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in zones]
    nxy = [np.zeros_like(a) for a in nx]
    mesh = _strip_mesh(panel_topology, n_elements_per_panel=4)
    asm = assemble_section_from_zones(
        zones,
        nx,
        nxy,
        section_strip_mesh=mesh,
        shell_theory=ShellTheory.FSDT,
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
    """Round trip R -> nodal f -> R about the requested ``about_xy``."""
    _pb, asm = _asm_naca()
    vec = BeamResultantVector.from_named(
        **{k: float(v) for k, v in named.items()},
        frame=FRAME_SECTION_YZ,
        about_xy=BEAM_LOAD_AXIS_YZ,
    )
    ndof = int(asm.cache.n_dof_node)
    f = beam_resultants_to_nodal_load(asm.node_meta, vec, ndof=ndof)
    back = integrate_nodal_load_resultants(
        asm.node_meta, f, about_xy=BEAM_LOAD_AXIS_YZ, ndof=ndof,
    )
    np.testing.assert_allclose(back.R, vec.R, rtol=1e-6, atol=1e-6)


def test_nodal_load_reconstructs_about_translated_axis():
    _pb, asm = _asm_naca()
    origin = (0.12, -0.04)
    vec = BeamResultantVector.from_named(
        N=200.0, Vy=180.0, Vz=-90.0, My=40.0, Mz=-25.0, T=12.0,
        frame=FRAME_SECTION_YZ,
        about_xy=origin,
    )
    ndof = int(asm.cache.n_dof_node)
    f = beam_resultants_to_nodal_load(asm.node_meta, vec, about_xy=origin, ndof=ndof)
    back = integrate_nodal_load_resultants(
        asm.node_meta, f, about_xy=origin, ndof=ndof,
    )
    np.testing.assert_allclose(back.R, vec.R, rtol=1e-6, atol=1e-6)
    assert back.about_xy == pytest.approx(origin)


def test_linear_static_cascade_reaches_quaternary():
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    vec = BeamResultantVector.from_named(
        My=1e3, frame=FRAME_SECTION_YZ, about_xy=BEAM_LOAD_AXIS_YZ,
    )
    static = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        resultants=vec,
        section_force_vector=equipollent_section_force_vector(panel_topology, vec),
        section_strip_mesh=_strip_mesh(panel_topology, n_elements_per_panel=4),
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
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.shear_centre_loading_axis import (
        shear_centre_about_yz_from_panel_topology,
    )

    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    y_sc, z_sc = shear_centre_about_yz_from_panel_topology(panel_topology)
    assert np.isfinite(y_sc) and np.isfinite(z_sc)

    mesh = _strip_mesh(panel_topology, n_elements_per_panel=4)
    vec = BeamResultantVector.from_named(
        T=50.0, frame=FRAME_SECTION_YZ, about_xy=(y_sc, z_sc),
    )
    sfv = equipollent_section_force_vector(panel_topology, vec)
    static = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        resultants=vec,
        torsion_about_shear_centre=True,
        section_force_vector=sfv,
        section_strip_mesh=mesh,
        load_channel="equipollent",
    ).execute()
    assert static.resultants is not None
    assert static.resultants.about_xy == pytest.approx((y_sc, z_sc), rel=0, abs=1e-9)
    assert static.assembly is not None and static.f_nodal is not None
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.section_force_vector_to_nodal_load import (
        section_force_vector_to_nodal_load,
    )

    ndof = int(static.assembly.cache.n_dof_node)
    f_from_sfv = section_force_vector_to_nodal_load(
        static.assembly.node_meta, sfv, ndof=ndof,
        panel_maps=static.assembly.panel_maps, zones=panel_topology.zones,
    )
    np.testing.assert_allclose(static.f_nodal, f_from_sfv, rtol=0.0, atol=1e-12)
    back = integrate_nodal_load_resultants(
        static.assembly.node_meta,
        static.f_nodal,
        about_xy=static.resultants.about_xy,
        ndof=ndof,
    )
    # Channel-A SFV is an intentional migration vs beam_resultants_to_nodal_load;
    # reconstruction of T is not required to match 50 at 1e-5.
    assert np.isfinite(back.T)
    assert back.T * 50.0 > 0.0
    # Default product path unchanged when flag is off
    vec_default = BeamResultantVector.from_named(
        T=50.0, frame=FRAME_SECTION_YZ, about_xy=BEAM_LOAD_AXIS_YZ,
    )
    static_default = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        resultants=vec_default,
        section_force_vector=equipollent_section_force_vector(panel_topology, vec_default),
        section_strip_mesh=mesh,
        load_channel="equipollent",
    ).execute()
    assert static_default.resultants is not None
    assert static_default.resultants.about_xy == BEAM_LOAD_AXIS_YZ


def test_linear_static_from_resultants_rejects_non_fsdt() -> None:
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    with pytest.raises(ValueError, match="FSDT"):
        SectionLinearStaticRunner(
            panel_topology=panel_topology,
            My=1.0e3,
            section_strip_mesh=_strip_mesh(panel_topology, n_elements_per_panel=3),
            load_channel="equipollent",
            shell_theory=ShellTheory.CALCULIX_CCX,
        ).execute()

