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
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import (
    ShellPanelResultants,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.results.compute_tertiary import (
    FSDTInterfaceShearRecoveryOperator,
    ShellTertiaryResultsOrchestrator,
    fsdt_interface_shear_profile,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.results.compute_tertiary.recovery_operators import (
    _ply_transverse_g,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.results.containers.container_hopper import (
    ShellSecondaryResultSet,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.results.containers.elemental_results import (
    ShellElementalResults,
)


def _ply(**overrides) -> OrthotropicPly:
    base = dict(
        name="gfrp", E1=30e9, E2=8e9, G12=3e9, nu12=0.3, rho=1.5e3, t_ply=1.25e-4,
        Xt=500e6, Xc=300e6, Yt=50e6, Yc=200e6, S12=40e6, Zt=1e6, S13=6e6, S23=6e6,
    )
    base.update(overrides)
    return OrthotropicPly(**base)


def _reference_profile(*, qx, qy, z_iface, h, g13_ply, g23_ply):
    """Old per-ply Python-loop implementation, kept only as a reference oracle."""
    z = np.asarray(z_iface, dtype=np.float64).ravel()
    n_iface = z.size
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

    def _parabolic_weight(zi, h):
        xi = 2.0 * zi / h
        return max(0.0, 1.0 - xi * xi)

    w = np.array([_parabolic_weight(zi, h) for zi in z])

    def _int_parab(za, zb):
        return (zb - za) - (4.0 / (3.0 * h * h)) * (zb**3 - za**3)

    norm13 = norm23 = 0.0
    for k in range(n_ply):
        dz_int = _int_parab(float(z[k]), float(z[k + 1]))
        norm13 += float(g13[k]) * dz_int
        norm23 += float(g23[k]) * dz_int

    if norm13 > 0.0:
        out[:, 0] = qx * g13_i * w / norm13
    if norm23 > 0.0:
        out[:, 1] = qy * g23_i * w / norm23
    out[0, :] = 0.0
    out[-1, :] = 0.0
    return out


class TestFsdtInterfaceShearProfileVectorised:
    @pytest.mark.parametrize("n_ply", [1, 2, 3, 8, 64, 80])
    def test_matches_reference_loop(self, n_ply):
        rng = np.random.default_rng(n_ply)
        h = 0.01
        z_iface = np.linspace(-h / 2, h / 2, n_ply + 1)
        g13 = rng.uniform(1e9, 5e9, size=n_ply)
        g23 = rng.uniform(1e9, 5e9, size=n_ply)
        qx, qy = 1234.0, -567.0

        got = fsdt_interface_shear_profile(qx=qx, qy=qy, z_iface=z_iface, h=h, g13_ply=g13, g23_ply=g23)
        ref = _reference_profile(qx=qx, qy=qy, z_iface=z_iface, h=h, g13_ply=g13, g23_ply=g23)
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)

    def test_default_uniform_g_still_matches_reference(self):
        """No g13_ply/g23_ply supplied -> profile() defaults to ones(n_ply)."""
        h = 0.005
        z_iface = np.linspace(-h / 2, h / 2, 9)  # 8 plies
        qx, qy = 500.0, 300.0
        got = fsdt_interface_shear_profile(qx=qx, qy=qy, z_iface=z_iface, h=h)
        ref = _reference_profile(
            qx=qx, qy=qy, z_iface=z_iface, h=h,
            g13_ply=np.ones(8), g23_ply=np.ones(8),
        )
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


class TestOrchestratorHoistsPlyTransverseGOncePerPanel:
    def test_ply_transverse_g_called_once_per_panel_not_per_row(self):
        lam = LaminateDefinition(plies=[(_ply(), 0.0), (_ply(), 45.0), (_ply(), -45.0)])
        n_rows = 12  # all in the same panel -> should be exactly 1 call, not 12
        rows = [
            ShellPanelResultants(
                Nx=100.0, Ny=0.0, Nxy=0.0, Mx=0.0, My=0.0, Mxy=0.0,
                Qx=float(i + 1), Qy=float(-i - 1), thickness_m=3 * lam.plies[0][0].t_ply,
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
            "blade_structure_utilities.fem_utilities.section_shell_model."
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
                Nx=0.0, Ny=0.0, Nxy=0.0, Mx=0.0, My=0.0, Mxy=0.0,
                Qx=float(200 + 10 * i), Qy=float(-100 - 5 * i),
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
                qx=float(row.Qx), qy=float(row.Qy), lam=lam, thickness_m=float(row.thickness_m),
            )
            np.testing.assert_allclose(
                tertiary.ply_results.tau_iface[i, : expected.shape[0]], expected, rtol=1e-10, atol=1e-10
            )
