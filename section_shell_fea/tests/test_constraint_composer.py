"""WP4 composer: one constraint set for production and diagnostic modes."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.compose import (
    build_section_constraint_set,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.types import (
    constraint_equations_to_solver_dict,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.section_assembly import (
    assemble_section_from_zones,
)


pytestmark = pytest.mark.unit


def test_assembly_is_constraint_free_and_f_element_is_zero(assembled_5df):
    assert assembled_5df.node_meta.constraints in (None, {})
    assert float(np.linalg.norm(assembled_5df.f_element)) < 1e-14
    assert assembled_5df.n_scatter_add == 1


@pytest.mark.parametrize("tie", ["skin_offset", "none"])
def test_composer_shared_modes_are_deterministic(assembled_5df, tie):
    cs = build_section_constraint_set(
        assembled_5df,
        interface_constraint_mode="shared",
        cap_tie_mode=tie,
        bc_mode="contour_support",
    )
    again = build_section_constraint_set(
        assembled_5df,
        interface_constraint_mode="shared",
        cap_tie_mode=tie,
        bc_mode="contour_support",
    )
    assert constraint_equations_to_solver_dict(cs.equations) == (
        constraint_equations_to_solver_dict(again.equations)
    )
    assert cs.fixed_dofs == again.fixed_dofs
    if tie == "skin_offset":
        assert cs.report.n_cap_links > 0
        assert cs.report.n_cap_tie_equations > 0
    else:
        assert cs.report.n_cap_links == 0
        assert cs.report.n_cap_tie_equations == 0


def test_production_shared_skin_offset_has_ties_and_supports(assembled_5df):
    cs = build_section_constraint_set(
        assembled_5df,
        interface_constraint_mode="shared",
        cap_tie_mode="skin_offset",
        bc_mode="contour_support",
    )
    assert cs.report.interface_constraint_mode == "shared"
    assert cs.report.n_junction_equations == 0
    assert len(cs.fixed_dofs) > 0
    slaves = {eq.slave_dof for eq in cs.equations}
    assert slaves.isdisjoint(cs.fixed_dofs)


def test_composer_drops_mpc_slaves_already_in_fixed_dofs(assembled_5df):
    cs = build_section_constraint_set(
        assembled_5df,
        interface_constraint_mode="shared",
        cap_tie_mode="skin_offset",
        bc_mode="contour_support",
    )
    slaves = {eq.slave_dof for eq in cs.equations}
    assert slaves.isdisjoint(cs.fixed_dofs)
    assert len(cs.equations) > 0
    assert len(cs.fixed_dofs) > 0


def test_hybrid_collinear_mpc_clusters_drop_collinear_flags():
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.compose import (
        _mpc_clusters,
    )

    class _Topo:
        clusters = ["keep_non_collinear", "drop_collinear"]
        cluster_is_collinear_flags = {0: False, 1: True}
        panel_maps = []

    kept = _mpc_clusters(_Topo(), interface_constraint_mode="hybrid_collinear")
    assert kept == ["keep_non_collinear"]


def test_transformed_basis_composer_adds_junction_equations(assembled_5df):
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (
        build_section_view,
        resolve_system_type,
    )
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import (
        AirfoilSDF,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh import (
        build_section_strip_mesh,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff._section_to_shell_inputs import (
        shell_mesh_inputs_from_section,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
        shell_panel_topology_from_shell_inputs,
    )

    twist = float(np.deg2rad(34.1))
    airfoil = AirfoilSDF.from_naca_series(
        6, 63.0, 4.0, 15.0, n_points=200, chord=0.241, closed_te=True
    )
    section = build_section_view(
        airfoil,
        resolve_system_type("5D-F"),
        twist_angle_rad=twist,
        skin_thickness=0.001,
        web_thickness=0.0015,
        cap_height=0.004,
    )
    shell = shell_mesh_inputs_from_section(
        section, twist_rad=twist, layout_key="5D-F", n_web_samples=20, n_cap_samples=80
    )
    mesh = build_section_strip_mesh(shell, target_element_length_frac=0.012, endpoint_tol=1e-6)
    zones = list(shell_panel_topology_from_shell_inputs(shell, allow_demo_bind=True).zones)
    nx = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in zones]
    nxy = [np.zeros_like(v) for v in nx]
    asm = assemble_section_from_zones(
        zones,
        nx,
        nxy,
        L_x=1.0,
        interface_constraint_mode="transformed_basis",
        section_strip_mesh=mesh,
    )
    assert asm.node_meta.constraints in (None, {})
    cs = build_section_constraint_set(
        asm,
        interface_constraint_mode="transformed_basis",
        cap_tie_mode="skin_offset",
    )
    assert cs.report.n_junction_equations > 0
    assert cs.report.n_cap_tie_equations >= 0
    assert len(cs.fixed_dofs) > 0
