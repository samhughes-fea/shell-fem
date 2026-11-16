import numpy as np
import pytest

from conftest import navier_clpt_centre_moments, simply_supported
from shell_fem import (
    DOF,
    BoundaryConditions,
    FEAMesh,
    FormulationField,
    FSDTModel,
    HashinCriterion,
    HygrothermalState,
    Laminate,
    LoadCase,
    RESULTS_CSV_SCHEMA,
    ResultsExtractor,
    StructuralGlobalSolver,
    integrate_abd,
    extrapolation_matrix,
)

A = 0.2


@pytest.fixture(scope="module")
def plate(request):
    from shell_fem import create_default_library

    cfrp = create_default_library().get("T300/5208")
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], 0.5e-3)
    mesh = FEAMesh.rectangular_plate(A, A, 16, 16, lam, FSDTModel())
    loads = LoadCase().add_pressure(1.0e3)
    result = StructuralGlobalSolver().solve(mesh, simply_supported(mesh, A, A), loads)
    return lam, mesh, result


def test_extrapolation_is_exact_for_bilinear_fields():
    g = 1 / np.sqrt(3)
    pts = [(-g, -g), (-g, g), (g, -g), (g, g)]
    f = lambda x, e: 2.0 + 3.0 * x - 1.5 * e + 0.7 * x * e
    corners = extrapolation_matrix(pts) @ np.array([f(*p) for p in pts])
    assert np.allclose(corners, [f(-1, -1), f(1, -1), f(1, 1), f(-1, 1)])
    assert np.allclose(extrapolation_matrix([(0.0, 0.0)]), 1.0)


def test_energy_split_and_global_balance(plate):
    lam, mesh, result = plate
    totals = ResultsExtractor(result).totals()
    assert sum(totals.energy_split.values()) == pytest.approx(totals.strain_energy, rel=1e-10)
    assert totals.strain_energy == pytest.approx(totals.external_work, rel=1e-8)  # mechanical load only
    assert totals.equilibrium_residual < 1e-10
    assert totals.applied_resultant[2] == pytest.approx(1.0e3 * A * A)
    assert totals.reaction_resultant[2] == pytest.approx(-1.0e3 * A * A)
    assert totals.energy_split["bending"] > 0.9 * totals.strain_energy
    assert totals.max_displacement["UZ"][1] == mesh.closest_node((A / 2, A / 2))
    expected_mass = sum(e.laminate.areal_density() * e.area for e in mesh.elements)
    assert totals.mass == pytest.approx(expected_mass)
    assert totals.mass == pytest.approx(lam.areal_density() * A * A)


def test_centre_moments_at_every_level_match_navier(plate):
    lam, mesh, result = plate
    reference = navier_clpt_centre_moments(integrate_abd(lam)[3:, 3:], A, A, 1.0e3)
    ex = ResultsExtractor(result)
    centre = mesh.closest_node((A / 2, A / 2))
    nodes = ex.nodes()
    nodal = nodes.M[list(nodes.node_ids).index(centre), :2]
    gp = ex.gauss_points()
    nearest = np.argsort(np.linalg.norm(gp.xyz[:, :2] - [A / 2, A / 2], axis=1))[:4]
    gauss = gp.M[nearest, :2].mean(axis=0)
    el = ex.elements()
    nearest_el = np.argsort(np.linalg.norm(el.centroid[:, :2] - [A / 2, A / 2], axis=1))[:4]
    element = el.M[nearest_el, :2].mean(axis=0)
    for level in (nodal, gauss, element):
        assert np.allclose(level, reference, rtol=0.03)
    assert nodes.element_count[list(nodes.node_ids).index(centre)] == 4
    assert nodes.element_count[0] == 1


def test_membrane_resultant_is_uniform(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 45, -45, 90, 90, -45, 45, 0], 0.125e-3)
    L, b = 0.1, 0.04
    mesh = FEAMesh.rectangular_plate(L, b, 5, 2, lam, FSDTModel())
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UZ, DOF.RX, DOF.RY])
    bcs.fix(mesh.nodes_on_line(0, 0.0), [DOF.UX]).fix([mesh.closest_node((0, 0))], [DOF.UY])
    loads = LoadCase()
    right = sorted(mesh.nodes_on_line(0, L), key=lambda n: mesh.node_coordinates(n)[1])
    for i, node in enumerate(right):
        loads.add_nodal_load(node, DOF.UX, 2.0e4 * b / 2 * (0.5 if i in (0, len(right) - 1) else 1.0))
    ex = ResultsExtractor(StructuralGlobalSolver().solve(mesh, bcs, loads))
    for field in (ex.gauss_points(), ex.elements(), ex.nodes()):
        assert np.allclose(field.N, [2.0e4, 0.0, 0.0], atol=1e-6 * 2.0e4)


def test_thermal_resultants_are_mechanical(aluminium):
    lam = Laminate.from_angles(aluminium, [0], 1e-3)
    mesh = FEAMesh.rectangular_plate(0.1, 0.1, 3, 3, lam, FSDTModel())
    c0, c1, c2 = mesh.closest_node((0, 0)), mesh.closest_node((0.1, 0)), mesh.closest_node((0, 0.1))
    bcs = BoundaryConditions().fix([c0], [0, 1, 2]).fix([c1], [1, 2]).fix([c2], [2])
    result = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().set_hygrothermal(HygrothermalState(80.0)))
    ex = ResultsExtractor(result)
    assert np.allclose(ex.gauss_points().N, 0.0, atol=1e-3)
    assert ex.totals().equilibrium_residual < 1e-9
    pp = ex.ply_points()
    assert np.allclose(pp.strain_mechanical, 0.0, atol=1e-10)
    alpha = aluminium.hygrothermal.alpha1
    assert pp.strain_global[:, :2] == pytest.approx(alpha * 80.0, rel=1e-8, abs=1e-12)


def test_ply_fields_failure_and_nodal_stresses(plate):
    lam, mesh, result = plate
    ex = ResultsExtractor(result, criterion=HashinCriterion("3d"))
    pp = ex.ply_points()
    assert len(pp.z) == mesh.n_elements * 4 * lam.n_plies * 3
    assert pp.strain_global.shape == pp.stress_global.shape == (pp.z.size, 3)
    assert pp.strain_mechanical.shape == pp.strain_material.shape == (pp.z.size, 3)
    assert np.allclose(pp.strain_global, pp.strain_mechanical)
    for i in (0, pp.z.size // 2, pp.z.size - 1):
        ply = lam.plies[int(pp.ply_index[i])]
        assert ply.Q_bar @ pp.strain_mechanical[i] == pytest.approx(pp.stress_global[i], rel=1e-10, abs=1e-8)
        assert pp.strain_material[i] == pytest.approx(ply.to_material_strain(pp.strain_mechanical[i]))
    assert np.all(np.isfinite(pp.failure_index))
    totals = ex.totals()
    assert totals.max_failure[0] == pytest.approx(np.max(pp.failure_index))
    el = ex.elements()
    assert np.nanmax(el.max_failure_index) == pytest.approx(totals.max_failure[0])
    node_ids, s = ex.nodal_ply_stress(ply=0, position="bottom", frame="material")
    centre = list(node_ids).index(mesh.closest_node((A / 2, A / 2)))
    assert s.shape == (mesh.n_nodes, 5)
    assert s[centre, 0] < 0.0  # upward pressure arches the plate: the bottom 0-degree ply is compressed
    with pytest.raises(ValueError):
        ex.nodal_ply_stress(0, frame="bogus")


def test_formulation_operators_and_section_matrices(plate):
    lam, mesh, result = plate
    ex = ResultsExtractor(result)
    form, gp = ex.formulation(), ex.gauss_points()
    assert isinstance(form, FormulationField)
    assert form.element_ids.shape == gp.element_ids.shape
    assert np.array_equal(form.element_ids, gp.element_ids)
    assert np.array_equal(form.point_index, gp.point_index)
    assert np.allclose(form.natural, gp.natural)
    assert np.allclose(form.xyz, gp.xyz)
    assert np.allclose(form.weight * form.detJ, gp.weight)
    assert form.N.ndim == 2 and np.allclose(form.N.sum(axis=1), 1.0)
    assert form.B_membrane.shape[1:] == (3, int(form.n_dof[0]))
    assert form.B_bending.shape[1:] == (3, int(form.n_dof[0]))
    assert form.B_shear.shape[1:] == (2, int(form.n_dof[0]))
    for i, element in enumerate(mesh.shell_elements):
        for j, (xi, eta, w) in enumerate(element.recovery_rule()):
            k = i * 4 + j
            J, detJ, _ = element.jacobian(xi, eta)
            N, _ = element.shape_functions(xi, eta)
            B_m, B_b, B_s = element.strain_operators(xi, eta)
            assert form.element_ids[k] == element.element_id
            assert form.weight[k] == pytest.approx(w)
            assert form.detJ[k] == pytest.approx(detJ)
            assert np.allclose(form.jacobian[k], J)
            assert np.allclose(form.N[k], N)
            assert np.allclose(form.B_membrane[k], B_m)
            assert np.allclose(form.B_bending[k], B_b)
            assert np.allclose(form.B_shear[k], B_s)
            assert np.allclose(form.C_ee[k], element.section.C_ee)
            assert np.allclose(form.C_ss[k], element.section.C_ss)
            NM = form.C_ee[k][:6] @ np.concatenate([gp.eps0[k], gp.kappa[k]])
            Q = form.C_ss[k][:2] @ gp.gamma0[k]
            assert NM[:3] == pytest.approx(gp.N[k], rel=1e-10, abs=1e-8)
            assert NM[3:6] == pytest.approx(gp.M[k], rel=1e-10, abs=1e-8)
            assert Q == pytest.approx(gp.Q[k], rel=1e-10, abs=1e-8)


def test_csv_export(plate, tmp_path):
    lam, mesh, result = plate
    paths = ResultsExtractor(result).to_csv(tmp_path, include_plies=False)
    assert set(paths) == {"gauss_points", "elements", "nodes", "totals", "formulation"}
    schema = f"# schema={RESULTS_CSV_SCHEMA}"
    for path in paths.values():
        assert path.read_text().splitlines()[0] == schema
    header = paths["nodes"].read_text().splitlines()[1].split(",")
    assert header[:3] == ["node_ids", "xyz_0", "xyz_1"]
    assert len(paths["gauss_points"].read_text().splitlines()) == mesh.n_elements * 4 + 2
    assert len(paths["formulation"].read_text().splitlines()) == mesh.n_elements * 4 + 2
    form_header = paths["formulation"].read_text().splitlines()[1]
    assert "detJ" in form_header
    assert "weight" in form_header
    assert "jacobian_0" in form_header
    totals = dict(line.split(",") for line in paths["totals"].read_text().splitlines()[2:])
    assert float(totals["mass"]) == pytest.approx(ResultsExtractor(result).totals().mass)
    ply_paths = ResultsExtractor(result).to_csv(tmp_path / "plies", include_plies=True)
    ply_header = ply_paths["ply_points"].read_text().splitlines()[1]
    assert "strain_global_0" in ply_header
    assert "strain_mechanical_0" in ply_header
    assert "strain_material_0" in ply_header
