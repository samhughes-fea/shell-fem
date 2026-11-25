"""ASME GCI and mesh convergence studies."""

import math
import warnings

import numpy as np
import pytest

from shell_fem import FEAMesh, FSDTModel, LoadCase
from shell_fem.pre_processing.load_library.blade_loads import add_sectional_force, add_sectional_torque
from shell_fem.processing.diagnostics import grid_convergence_index, mesh_size
from shell_fem.simulation_runner.convergence import MeshConvergenceStudy, blade_convergence_study
from shell_fem.verification.blade_benchmarks import L, tube_blade
from shell_fem.verification.shell_benchmarks import isotropic_laminate


@pytest.mark.parametrize("h, p", [((1.0, 2.0, 4.0), 2.0), ((1.0, 1.5, 2.4), 1.7), ((0.3, 0.45, 0.6), 1.0)])
def test_gci_recovers_the_order_and_limit_of_a_power_law(h, p):
    phi0, c = 2.0, 3.0
    values = [phi0 + c * hk ** p for hk in h]
    g = grid_convergence_index(h[::-1], values[::-1])      # any order in
    assert g.h == tuple(h) and g.fine == pytest.approx(values[0])
    assert g.order == pytest.approx(p, rel=1e-8)
    assert g.extrapolated == pytest.approx(phi0, rel=1e-8)
    r21 = h[1] / h[0]
    assert g.gci == pytest.approx(1.25 * abs((values[1] - values[0]) / values[0]) / (r21 ** p - 1.0), rel=1e-8)
    assert g.asymptotic_ratio == pytest.approx(1.0, rel=1e-6)
    low, high = g.band
    assert low < phi0 + 1.25 * c * h[0] ** p and high > values[0] > phi0
    assert not g.oscillatory and not g.converged and g.in_asymptotic_range()


def test_gci_flags_oscillation_and_convergence():
    g = grid_convergence_index([1.0, 2.0, 4.0], [1.0, 1.1, 0.7])
    assert g.oscillatory and g.order > 0.0 and g.gci > 0.0
    # a result that wanders before it settles: the differences shrink far faster than the mesh
    assert not grid_convergence_index([1.0, 1.5, 2.25], [-0.37494, -0.37465, -0.37458]).in_asymptotic_range()
    g = grid_convergence_index([1.0, 2.0, 4.0], [5.0, 5.0 + 1e-12, 5.0 - 1e-12])
    assert g.converged and g.gci == 0.0 and g.extrapolated == 5.0
    g = grid_convergence_index([1.0, 2.0, 4.0], [1e-15, -2e-15, 3e-15], atol=1e-12)
    assert g.converged
    g = grid_convergence_index([1.0, 2.0, 4.0], [1.0, 1.0, 1.5])
    assert math.isnan(g.order) and math.isnan(g.gci)
    with pytest.raises(ValueError):
        grid_convergence_index([1.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        grid_convergence_index([1.0, 2.0], [1.0, 2.0])


def test_gci_with_a_fixed_order():
    g = grid_convergence_index([1.0, 2.0, 4.0], [1.1, 1.4, 2.6], order=2.0, safety_factor=3.0)
    assert g.order == 2.0 and g.gci == pytest.approx(3.0 * (0.3 / 1.1) / 3.0)


def test_mesh_size_of_a_plate():
    lam = isotropic_laminate(70e9, 0.3, 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = FEAMesh.rectangular_plate(2.0, 1.0, 8, 4, lam, FSDTModel())
    assert mesh_size(mesh) == pytest.approx(0.25)


def test_study_sorts_levels_and_reports_the_gci():
    lam = isotropic_laminate(70e9, 0.3, 0.01)

    def analyse(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = FEAMesh.rectangular_plate(1.0, 1.0, n, n, lam, FSDTModel())
        return mesh, {"q": 1.0 + (1.0 / n) ** 2, "zero": 0.0}

    results = MeshConvergenceStudy([16, 4, 8], analyse).execute().get_results()
    assert [lv.level for lv in results] == [4, 8, 16]
    np.testing.assert_allclose(results.h, [0.25, 0.125, 0.0625])
    g = results.gci("q")
    assert g.order == pytest.approx(2.0) and g.extrapolated == pytest.approx(1.0)
    assert results.gci("zero").converged
    assert set(results.summary()) == {"q", "zero"}
    assert "GCI fine" in results.table() and results.reliable("q") and results.reliable("zero")
    with pytest.raises(ValueError):
        MeshConvergenceStudy([4, 8], analyse)


def test_tube_ladder_brackets_beam_theory():
    """Curved Quad8 on a tube: the GCI band of the tip twist holds the beam value mL^2 / 2GJ."""
    from shell_fem.verification.blade_benchmarks import E, NU, R, T

    m = 1000.0

    def loads(blade):
        return [add_sectional_torque(add_sectional_force(LoadCase("c"), blade, lambda z: (0.0, 1000.0, 0.0)),
                                     blade, lambda z: m)]

    study = blade_convergence_study(lambda n: tube_blade("quad4", 2 * n, 2 * n), loads, "c", [4, 6, 9],
                                    atol={"tip_edge": 1e-9, "root_My": 1e-3})
    results = study.execute().get_results()
    twist = results.gci("tip_twist")
    GJ = 2.0 * np.pi * R ** 3 * T * E / (2.0 * (1.0 + NU))
    exact = m * L ** 2 / (2.0 * GJ)
    assert 1.0 < twist.order < 4.0 and not twist.oscillatory
    low, high = twist.band
    assert low <= exact * 1.01 and abs(twist.extrapolated / exact - 1.0) < abs(twist.fine / exact - 1.0)
    assert results.gci("root_Mx").converged and results.gci("tip_edge").converged


def test_reliable_needs_consistent_orders():
    h = [0.4, 0.3, 0.2, 0.15]

    def study(fn):
        return MeshConvergenceStudy(h, lambda hk: (_Mesh(hk), {"q": fn(hk)}), size=lambda m: m.h).execute().get_results()

    assert study(lambda x: 1.0 + x ** 2).reliable("q")
    # second order on the coarse meshes, fourth on the fine ones: the orders disagree
    mixed = study(lambda x: 1.0 + (x ** 2 if x > 0.25 else 0.0625 * (x / 0.25) ** 4))
    assert not mixed.reliable("q")


class _Mesh:
    def __init__(self, h):
        self.h, self.n_dof = h, int(1.0 / h ** 2)
