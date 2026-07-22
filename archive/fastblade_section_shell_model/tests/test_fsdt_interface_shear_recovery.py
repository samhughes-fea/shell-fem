"""FSDT interface transverse-shear recovery (tier 3 modular operator)."""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.materials.laminate import LaminateDefinition
from blade_precompute.materials.ply import OrthotropicPly
from blade_precompute.section_shell_model.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import (
    ShellPanelResultants,
)
from blade_precompute.section_shell_model.processing.results.compute_tertiary import (
    FSDTInterfaceShearRecoveryOperator,
    ShellTertiaryResultsOrchestrator,
    fsdt_interface_shear_profile,
)
from blade_precompute.section_shell_model.processing.results.containers.container_hopper import (
    ShellSecondaryResultSet,
)
from blade_precompute.section_shell_model.processing.results.containers.elemental_results import (
    ShellElementalResults,
)


def _ply(t_ply: float = 2e-3) -> OrthotropicPly:
    return OrthotropicPly(
        name="gfrp",
        E1=30e9,
        E2=8e9,
        G12=3e9,
        nu12=0.3,
        rho=1.5e3,
        t_ply=t_ply,
        Xt=500e6,
        Xc=300e6,
        Yt=50e6,
        Yc=200e6,
        S12=40e6,
        Zt=1e6,
        S13=6e6,
        S23=6e6,
    )


def test_parabolic_peak_matches_1p5_q_over_t_single_ply_uniform():
    """Homogeneous single-ply: free surfaces zero; midplane peak factor is 1.5 Q/t."""
    h = 2e-3
    qx, qy = 300.0, 400.0
    z_surf = np.array([-0.5 * h, 0.5 * h])
    g1 = np.array([1.0])
    tau_surf = fsdt_interface_shear_profile(
        qx=qx, qy=qy, z_iface=z_surf, h=h, g13_ply=g1, g23_ply=g1,
    )
    np.testing.assert_allclose(tau_surf, 0.0)

    # Probe midplane with a uniform 2-interface split of the same total thickness
    # (mathematically identical continuous parabola / norm for constant G).
    z_probe = np.array([-0.5 * h, 0.0, 0.5 * h])
    g2 = np.array([1.0, 1.0])
    tau = fsdt_interface_shear_profile(
        qx=qx, qy=qy, z_iface=z_probe, h=h, g13_ply=g2, g23_ply=g2,
    )
    assert tau[1, 0] == pytest.approx(1.5 * qx / h)
    assert tau[1, 1] == pytest.approx(1.5 * qy / h)
    assert float(np.hypot(tau[1, 0], tau[1, 1])) == pytest.approx(1.5 * np.hypot(qx, qy) / h)

    # Operator path on a true single-ply laminate: n_iface = 2, surfaces only.
    lam = LaminateDefinition(plies=[(_ply(h), 0.0)])
    op = FSDTInterfaceShearRecoveryOperator()
    tau_op = op.recover_interfaces(qx=qx, qy=qy, lam=lam, thickness_m=h)
    assert tau_op.shape == (2, 2)
    np.testing.assert_allclose(tau_op, 0.0)


def test_surface_tau_zero_and_interface_count_n_ply_plus_one():
    lam = LaminateDefinition(plies=[(_ply(1e-3), 0.0), (_ply(1e-3), 90.0), (_ply(1e-3), 0.0)])
    op = FSDTInterfaceShearRecoveryOperator()
    qx, qy = 100.0, -50.0
    tau = op.recover_interfaces(qx=qx, qy=qy, lam=lam)
    n_ply = len(lam.plies)
    assert tau.shape == (n_ply + 1, 2)
    np.testing.assert_allclose(tau[0], 0.0)
    np.testing.assert_allclose(tau[-1], 0.0)
    # Interior interfaces non-zero under load.
    assert np.any(np.abs(tau[1:-1]) > 0.0)


def test_tertiary_orchestrator_still_passes_old_transverse_shear():
    p = _ply(2e-4)
    lam = LaminateDefinition(plies=[(p, 0.0), (p, 0.0)])
    rows = [
        ShellPanelResultants(
            Nx=1e3, Ny=0.0, Nxy=0.0, Mx=0.0, My=0.0, Mxy=0.0,
            Qx=1.0, Qy=2.0, thickness_m=4e-4, panel_index=0, station_index=0,
        )
    ]
    cache = ShellFormulationCache(stiffness_objects=[], force_objects=[], panel_laminates=[lam])
    secondary = ShellSecondaryResultSet(elemental_results=ShellElementalResults(resultant_rows=rows))
    tertiary = ShellTertiaryResultsOrchestrator(secondary=secondary, cache=cache).compute()

    np.testing.assert_allclose(tertiary.transverse_shear[:, 0], 1.0)
    np.testing.assert_allclose(tertiary.transverse_shear[:, 1], 2.0)
    assert tertiary.ply_results is not None
    assert tertiary.ply_results.tau_iface is not None
    assert tertiary.ply_results.tau_iface.shape == (1, 3, 2)
    np.testing.assert_allclose(tertiary.ply_results.tau_iface[0, 0], 0.0)
    np.testing.assert_allclose(tertiary.ply_results.tau_iface[0, -1], 0.0)
    assert tertiary.meta.get("shear_recovery_mode") == "fsdt_interface_parabolic"
    assert tertiary.ply_results.shear_trusted_mask is not None
    assert bool(tertiary.ply_results.shear_trusted_mask[0])
