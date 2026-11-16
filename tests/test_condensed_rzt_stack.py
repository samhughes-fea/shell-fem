"""Condensed RZT beside FSDT: same 6-DOF mesh, exact and equilibrium through the stack.

``RefinedZigZagModel(condensed=True)`` slaves the zig-zag amplitude to the mean
shear, so a sandwich or thick cap keeps FSDT's six DOFs per node while using the
harmonic shear stiffness A_s = h G. Full RZT (independent ψ) is the Pagano path
in ``test_gap_closure``; this file is the 6-DOF variant on the same mesh as FSDT.
"""

import numpy as np
import pytest

from conftest import simply_supported
from shell_fem import (
    ElasticProperties3D,
    EquilibriumStressRecoverer,
    ExactStressRecoverer,
    FEAMesh,
    FSDTModel,
    HygrothermalProperties,
    Laminate,
    LoadCase,
    MaterialCard,
    Ply,
    RefinedZigZagModel,
    StrengthProperties3D,
    StructuralGlobalSolver,
    ply_stack_stress_strain,
)

A, Q, N_ELEM = 0.2, 1.0e4, 6
SKIN_T, CORE_T = 0.5e-3, 8.0e-3

FOAM = MaterialCard(
    name="PVC foam H100",
    elastic=ElasticProperties3D.isotropic(105e6, 0.3),
    strength=StrengthProperties3D(3.1e6, 1.9e6, 3.1e6, 1.9e6, 1.6e6),
    hygrothermal=HygrothermalProperties(alpha1=35e-6, alpha2=35e-6),
    density=100.0,
)


def sandwich(skin):
    return Laminate(
        [
            Ply(skin, SKIN_T, 0),
            Ply(FOAM, CORE_T, 0),
            Ply(skin, SKIN_T, 0),
        ],
        name="sandwich cap",
    )


def solve(lam, model):
    mesh = FEAMesh.rectangular_plate(A, A, N_ELEM, N_ELEM, lam, model)
    loads = LoadCase().add_pressure(Q)
    result = StructuralGlobalSolver().solve(mesh, simply_supported(mesh, A, A), loads)
    return mesh, loads, result


def centre_w(mesh, result):
    return result.node_displacement(mesh.closest_node((A / 2, A / 2)))[2]


@pytest.fixture(scope="module")
def sandwich_pair(cfrp):
    lam = sandwich(cfrp)
    fsdt = solve(lam, FSDTModel())
    crz = solve(lam, RefinedZigZagModel(condensed=True))
    return lam, fsdt, crz


def test_condensed_rzt_keeps_fsdt_dofs_on_the_same_mesh(sandwich_pair):
    lam, (mesh_fsdt, _, _), (mesh_crz, _, _) = sandwich_pair
    assert mesh_fsdt.n_nodes == mesh_crz.n_nodes
    assert mesh_fsdt.n_elements == mesh_crz.n_elements
    assert mesh_fsdt.n_dof == mesh_crz.n_dof
    for mesh in (mesh_fsdt, mesh_crz):
        element = mesh.elements[0]
        assert element.dofs_per_node == 6
        assert element.n_enrichment == 0
        assert element.laminate.n_plies == lam.n_plies


def test_condensed_rzt_is_more_flexible_than_fsdt_on_a_sandwich(sandwich_pair):
    """FSDT smears G h; condensed RZT uses the harmonic modulus, so the core controls."""
    lam, (mesh_fsdt, _, sol_fsdt), (mesh_crz, _, sol_crz) = sandwich_pair
    as_fsdt = FSDTModel().transverse_shear_matrix(lam)
    as_crz = RefinedZigZagModel(condensed=True).transverse_shear_matrix(lam)
    assert as_crz[0, 0] < 0.2 * as_fsdt[0, 0]
    w_fsdt, w_crz = centre_w(mesh_fsdt, sol_fsdt), centre_w(mesh_crz, sol_crz)
    assert w_crz > 1.3 * w_fsdt
    mesh_full, _, sol_full = solve(lam, RefinedZigZagModel())
    # condensed RZT is the 6-DOF sibling, not a substitute for independent ψ
    assert mesh_crz.n_dof < mesh_full.n_dof
    assert w_crz == pytest.approx(centre_w(mesh_full, sol_full), rel=0.15)


def test_ply_stack_exact_covers_every_ply_on_condensed_rzt(sandwich_pair):
    lam, _, (mesh, _, result) = sandwich_pair
    field = ply_stack_stress_strain(result)
    n_gp = mesh.n_elements * 4
    assert field.z.size == n_gp * lam.n_plies * 3
    assert set(field.ply_index.tolist()) == set(range(lam.n_plies))
    assert np.allclose(field.normal_stress, 0.0)
    # constitutive condensed RZT shear is continuous (tau = G gamma0 in every ply)
    rec = ExactStressRecoverer()
    element, u = mesh.elements[1], result.element_displacements(mesh.elements[1].element_id)
    for z in lam.z_interfaces[1:-1]:
        below = rec.recover(element, u, z, side="below").shear_global
        above = rec.recover(element, u, z, side="above").shear_global
        assert np.allclose(below, above, rtol=1e-10)
        assert np.linalg.norm(above) > 0.0


def test_ply_stack_equilibrium_recovers_sigma_z_and_continuous_shear(sandwich_pair):
    lam, _, (mesh, loads, result) = sandwich_pair
    rec = EquilibriumStressRecoverer(result, loads)
    field = ply_stack_stress_strain(result, recoverer=rec, load_case=loads)
    n_gp = mesh.n_elements * 4
    assert field.z.size == n_gp * lam.n_plies * 3
    top = field.ply_index == lam.n_plies - 1
    top_face = top & np.isclose(field.z, lam.z_interfaces[-1])
    assert np.allclose(field.normal_stress[top_face], Q, rtol=1e-9)
    bottom_face = (field.ply_index == 0) & np.isclose(field.z, lam.z_interfaces[0])
    assert np.allclose(field.normal_stress[bottom_face], 0.0, atol=1e-9)

    interior = mesh.elements[N_ELEM * N_ELEM // 2 + N_ELEM // 2]
    for i, (xi, eta, _) in enumerate(interior.gauss_points()):
        for k, z in enumerate(lam.z_interfaces[1:-1], start=1):
            below = (field.element_ids == interior.element_id) & (field.point_index == i)
            below = below & (field.ply_index == k - 1) & np.isclose(field.z, z)
            above = (field.element_ids == interior.element_id) & (field.point_index == i)
            above = above & (field.ply_index == k) & np.isclose(field.z, z)
            assert np.allclose(field.shear_global[below], field.shear_global[above], rtol=1e-10)
            tyz, txz, sz = rec.transverse_stresses(interior, xi, eta, float(z))
            assert field.normal_stress[below][0] == pytest.approx(sz, rel=1e-10, abs=1e-8)
            assert field.shear_global[below][0] == pytest.approx([tyz, txz], rel=1e-10, abs=1e-8)
