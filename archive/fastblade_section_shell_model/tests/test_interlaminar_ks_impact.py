"""Interlaminar KS impact after first-ply S13 + interface-τ policy.

Phase 4 aligned quaternary / optimiser sparse fallback on first-ply
``interlaminar_allowable_from_lam``. This gate locks the engineering effects the
next-steps polish table called out:

1. First-ply S13 is more conservative than a silent max-over-plies allowable when
   mid-stack plies are stronger.
2. Homogeneous multi-ply interface τ recovers the classical peak ``1.5 |Q|/t``.
3. Heterogeneous G redistributes interface peak so FI can diverge from the
   naive scalar envelope — KS then tracks the interface value, not the scalar.
"""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.materials.laminate import LaminateDefinition
from blade_precompute.materials.ply import OrthotropicPly
from blade_precompute.section_optimisation.engine.aggregation import ks_aggregate
from blade_precompute.section_shell_model.api.fi_recovery import (
    compute_quaternary_fi_from_panel_rows,
    interlaminar_allowable_from_lam,
    interlaminar_fi_from_row,
    interlaminar_fi_from_row_fsdt,
    max_interlaminar_fi_from_quaternary,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import (
    ShellPanelResultants,
)


def _ply(*, t_ply: float = 1e-3, S13: float = 6e6, G13: float = 3e9) -> OrthotropicPly:
    ply = OrthotropicPly(
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
        S13=S13,
        S23=S13,
    )
    # Transverse shear moduli are optional attributes used by FSDT interface recovery.
    ply.G13 = G13
    ply.G23 = G13
    return ply


def _row(*, qx: float = 200.0, qy: float = 0.0, t: float = 3e-3) -> ShellPanelResultants:
    return ShellPanelResultants(
        Nx=1e3,
        Ny=0.0,
        Nxy=0.0,
        Mx=0.0,
        My=0.0,
        Mxy=0.0,
        Qx=qx,
        Qy=qy,
        thickness_m=t,
        panel_index=0,
        station_index=0,
    )


def test_first_ply_s13_policy_raises_ks_vs_max_ply_allowable():
    """Weak outer / strong mid-stack: first-ply S13 must govern KS (more conservative)."""
    weak = _ply(S13=6e6)
    strong = _ply(S13=80e6)
    lam = LaminateDefinition(plies=[(weak, 0.0), (strong, 0.0), (weak, 0.0)])
    row = _row(qx=300.0, t=3e-3)

    s_first = interlaminar_allowable_from_lam(lam)
    s_max = max(float(p.S13) for p, _ in lam.plies)
    assert s_first == pytest.approx(6e6)
    assert s_max == pytest.approx(80e6)

    fi_first, nr_f, mode_f = interlaminar_fi_from_row_fsdt(row, lam, s_il=s_first)
    fi_max_ply, nr_m, mode_m = interlaminar_fi_from_row_fsdt(row, lam, s_il=s_max)
    assert not nr_f and not nr_m
    assert mode_f == mode_m == "interface"
    assert fi_first > fi_max_ply * 5.0  # S13 ratio ≈ 13×; interface τ equal

    # Quaternary bridge uses first-ply policy end-to-end.
    bundle = compute_quaternary_fi_from_panel_rows([[row]], [lam])
    fi_q = max_interlaminar_fi_from_quaternary(bundle)
    assert fi_q == pytest.approx(fi_first, rel=1e-9, abs=1e-12)

    rho = 50.0
    # Two-station synthetic envelope of identical interlaminar FI (typical KS input).
    ks_first = ks_aggregate(np.array([fi_first, fi_first]), rho)
    ks_max_ply = ks_aggregate(np.array([fi_max_ply, fi_max_ply]), rho)
    assert ks_first > ks_max_ply
    # Constraint form used by the optimiser: ``1 - KS`` (feasible when positive).
    assert (1.0 - ks_first) < (1.0 - ks_max_ply)


def test_homogeneous_interface_ks_matches_scalar_peak_envelope():
    """Homogeneous 2-ply: interface FI = scalar 1.5|Q|/t so KS envelopes match."""
    p = _ply(t_ply=1e-3, S13=6e6, G13=3e9)
    lam = LaminateDefinition(plies=[(p, 0.0), (p, 0.0)])
    row = _row(qx=240.0, qy=80.0, t=2e-3)
    s_il = interlaminar_allowable_from_lam(lam)

    fi_iface, nr_i, mode = interlaminar_fi_from_row_fsdt(row, lam, s_il=s_il)
    fi_scalar, nr_s = interlaminar_fi_from_row(row, s_il=s_il, peak_factor=1.5)
    assert not nr_i and not nr_s and mode == "interface"
    assert fi_iface == pytest.approx(fi_scalar, rel=1e-9, abs=1e-12)

    rho = 50.0
    assert ks_aggregate(np.array([fi_iface]), rho) == pytest.approx(
        ks_aggregate(np.array([fi_scalar]), rho),
        rel=1e-9,
        abs=1e-12,
    )


def test_heterogeneous_g_interface_ks_diverges_from_naive_scalar():
    """Heterogeneous G: interface peak ≠ 1.5|Q|/t; optimiser KS must track interface."""
    soft = _ply(t_ply=1e-3, S13=6e6, G13=1e9)
    stiff = _ply(t_ply=1e-3, S13=6e6, G13=10e9)
    lam = LaminateDefinition(plies=[(soft, 0.0), (stiff, 0.0), (soft, 0.0)])
    row = _row(qx=300.0, qy=0.0, t=3e-3)
    s_il = interlaminar_allowable_from_lam(lam)

    fi_iface, nr, mode = interlaminar_fi_from_row_fsdt(row, lam, s_il=s_il)
    fi_scalar, _ = interlaminar_fi_from_row(row, s_il=s_il, peak_factor=1.5)
    assert not nr and mode == "interface"
    assert abs(fi_iface - fi_scalar) > 1e-6 * max(fi_scalar, 1e-12)

    bundle = compute_quaternary_fi_from_panel_rows([[row]], [lam])
    fi_q = max_interlaminar_fi_from_quaternary(bundle)
    assert fi_q == pytest.approx(fi_iface, rel=1e-9, abs=1e-12)

    rho = 50.0
    ks_iface = ks_aggregate(np.array([fi_iface]), rho)
    ks_scalar = ks_aggregate(np.array([fi_scalar]), rho)
    assert abs(ks_iface - ks_scalar) > 1e-9
