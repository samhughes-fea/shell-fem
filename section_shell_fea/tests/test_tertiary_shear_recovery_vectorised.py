"""Regression tests for the tertiary transverse-shear recovery performance fix.

Two changes, both purely for performance (no intended change in FI/stress
values), mirroring the earlier ply_hashin_fi_modes fix:

1. fsdt_interface_shear_profile's three per-ply Python loops (interface-G
   averaging, parabolic weights, normalisation integrals) are now vectorised
   numpy expressions.
2. ShellTertiaryResultsOrchestrator computes _ply_transverse_g(lam) once per
   panel and passes it through, instead of every row in the panel
   recomputing it from scratch (every row in a panel shares the same
   laminate, so the per-ply G13/G23 values are identical across all of them).
"""

from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pytest

from blade_structure_utilities.material_utilities.laminate import LaminateDefinition
from blade_structure_utilities.material_utilities.orthotropic import OrthotropicPly
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellPanelResultants,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_tertiary import (
    FSDTInterfaceShearRecoveryOperator,
    ShellTertiaryResultsOrchestrator,
    fsdt_interface_shear_profile,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_tertiary.recovery_operators import (
    _ply_transverse_g,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.container_hopper import (
    ShellSecondaryResultSet,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.elemental_results import (
    ShellElementalResults,
)


def _ply(**overrides) -> OrthotropicPly:
    base = dict(
        name="gfrp", E1=30e9, E2=8e9, G12=3e9, nu12=0.3, rho=1.5e3, t_ply=1.25e-4,
        Xt=500e6, Xc=300e6, Yt=50e6, Yc=200e6, S12=40e6, Zt=1e6, S13=6e6, S23=6e6,
    )
    base.update(overrides)
    return OrthotropicPly(**base)


def _reference_profile(*, q1, q2, x3_iface, h, g13_ply, g23_ply):
    """Old per-ply Python-loop implementation, kept only as a reference oracle."""
    x3 = np.asarray(x3_iface, dtype=np.float64).ravel()
    n_iface = x3.size
    out = np.zeros((n_iface, 2), dtype=np.float64)
    if h <= 0.0 or n_iface < 2:
        return out
    n_ply = n_iface - 1
    g13 = np.asarray(g13_ply, dtype=np.float64).ravel()
    g23 = np.asarray(g23_ply, dtype=np.float64).ravel()

    g13_i = np.zeros(n_iface, dtype=np.float64)
    g23_i = np.zeros(n_iface, dtype=np.float64)
    g13_i[0], g13_i[-1] = g13[0], g13[-1]
    g23_i[0], g23_i[-1] = g23[0], g23[-1]
    for i in range(1, n_iface - 1):
        g13_i[i] = 0.5 * (g13[i - 1] + g13[i])
        g23_i[i] = 0.5 * (g23[i - 1] + g23[i])

    def _parabolic_weight(x3_i, h):
        xi = 2.0 * x3_i / h
        return max(0.0, 1.0 - xi * xi)

    w = np.array([_parabolic_weight(x3_i, h) for x3_i in x3])

    def _int_parab(x3_a, x3_b):
        return (x3_b - x3_a) - (4.0 / (3.0 * h * h)) * (x3_b**3 - x3_a**3)

    norm13 = norm23 = 0.0
    for k in range(n_ply):
        dx3_int = _int_parab(float(x3[k]), float(x3[k + 1]))
        norm13 += float(g13[k]) * dx3_int
        norm23 += float(g23[k]) * dx3_int

    if norm13 > 0.0:
        out[:, 0] = q1 * g13_i * w / norm13
    if norm23 > 0.0:
        out[:, 1] = q2 * g23_i * w / norm23
    out[0, :] = 0.0
    out[-1, :] = 0.0
    return out


class TestFsdtInterfaceShearProfileVectorised:
    @pytest.mark.parametrize("n_ply", [1, 2, 3, 8, 64, 80])
    def test_matches_reference_loop(self, n_ply):
        rng = np.random.default_rng(n_ply)
        h = 0.01
        x3_iface = np.linspace(-h / 2, h / 2, n_ply + 1)
        g13 = rng.uniform(1e9, 5e9, size=n_ply)
        g23 = rng.uniform(1e9, 5e9, size=n_ply)
        q1, q2 = 1234.0, -567.0

        got = fsdt_interface_shear_profile(q1=q1, q2=q2, x3_iface=x3_iface, h=h, g13_ply=g13, g23_ply=g23)
        ref = _reference_profile(q1=q1, q2=q2, x3_iface=x3_iface, h=h, g13_ply=g13, g23_ply=g23)
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)

    def test_default_uniform_g_still_matches_reference(self):
        """No g13_ply/g23_ply supplied -> profile() defaults to ones(n_ply)."""
        h = 0.005
        x3_iface = np.linspace(-h / 2, h / 2, 9)  # 8 plies
        q1, q2 = 500.0, 300.0
        got = fsdt_interface_shear_profile(q1=q1, q2=q2, x3_iface=x3_iface, h=h)
        ref = _reference_profile(
            q1=q1, q2=q2, x3_iface=x3_iface, h=h,
            g13_ply=np.ones(8), g23_ply=np.ones(8),
        )
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


class TestOrchestratorHoistsPlyTransverseGOncePerPanel:
    def test_ply_transverse_g_called_once_per_panel_not_per_row(self):
        lam = LaminateDefinition(plies=[(_ply(), 0.0), (_ply(), 45.0), (_ply(), -45.0)])
        n_rows = 12  # all in the same panel -> should be exactly 1 call, not 12
        rows = [
            ShellPanelResultants(
                N11=100.0, N22=0.0, N12=0.0, M11=0.0, M22=0.0, M12=0.0,
                Q1=float(i + 1), Q2=float(-i - 1), thickness_m=3 * lam.plies[0][0].t_ply,
                panel_index=0, station_index=i,
            )
            for i in range(n_rows)
        ]
        cache = ShellFormulationCache(stiffness_objects=[], force_objects=[], panel_laminates=[lam])
        secondary = ShellSecondaryResultSet(elemental_results=ShellElementalResults(resultant_rows=rows))

        call_count = 0
        real_fn = _ply_transverse_g

        def _counting(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return real_fn(*args, **kwargs)

        with mock.patch(
            "blade_structure_utilities.fem_utilities.section_shell_fea."
            "processing.static.results.compute_tertiary.tertiary_results_orchestrator._ply_transverse_g",
            side_effect=_counting,
        ):
            tertiary = ShellTertiaryResultsOrchestrator(secondary=secondary, cache=cache).compute()

        assert call_count == 1
        assert tertiary.ply_results is not None
        assert tertiary.ply_results.tau_iface is not None
        assert np.all(tertiary.ply_results.shear_trusted_mask)

    def test_hoisted_g13_g23_produce_identical_tau_iface_to_per_row_recompute(self):
        """Cross-check: orchestrator's hoisted-per-panel path must match calling
        recover_interfaces() fresh (no precomputed g13/g23) for every row."""
        lam = LaminateDefinition(plies=[(_ply(), 0.0), (_ply(), 90.0)])
        rows = [
            ShellPanelResultants(
                N11=0.0, N22=0.0, N12=0.0, M11=0.0, M22=0.0, M12=0.0,
                Q1=float(200 + 10 * i), Q2=float(-100 - 5 * i),
                thickness_m=2 * lam.plies[0][0].t_ply, panel_index=0, station_index=i,
            )
            for i in range(4)
        ]
        cache = ShellFormulationCache(stiffness_objects=[], force_objects=[], panel_laminates=[lam])
        secondary = ShellSecondaryResultSet(elemental_results=ShellElementalResults(resultant_rows=rows))
        tertiary = ShellTertiaryResultsOrchestrator(secondary=secondary, cache=cache).compute()

        op = FSDTInterfaceShearRecoveryOperator()
        for i, row in enumerate(rows):
            expected = op.recover_interfaces(
                q1=float(row.Q1), q2=float(row.Q2), lam=lam, thickness_m=float(row.thickness_m),
            )
            np.testing.assert_allclose(
                tertiary.ply_results.tau_iface[i, : expected.shape[0]], expected, rtol=1e-10, atol=1e-10
            )
