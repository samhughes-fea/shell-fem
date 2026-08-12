"""Unit tests for ASME V&V 20 / Roache GCI helpers."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.asme_gci import (
    FS_THREE_LEVEL,
    gci_fine,
    gci_triplet,
    observed_order,
    richardson_extrapolate,
)


def test_manufactured_second_order_recovers_p_and_small_gci():
    """Q = Qinf + C h^2 on constant-r ladder → p≈2, GCI small, Q_ext≈Qinf."""
    q_inf = 10.0
    c = 2.0
    h1, r = 0.01, 2.0
    h2, h3 = h1 * r, h1 * r * r
    q1 = q_inf + c * h1**2
    q2 = q_inf + c * h2**2
    q3 = q_inf + c * h3**2

    p, osc = observed_order(h1, h2, h3, q1, q2, q3)
    assert not osc
    assert p == pytest.approx(2.0, rel=1e-6)

    q_ext = richardson_extrapolate(h1, h2, q1, q2, p=p)
    assert q_ext == pytest.approx(q_inf, rel=1e-8)

    res = gci_triplet([h3, h2, h1], [q3, q2, q1])  # coarse→fine input ok
    assert res.ok
    assert not res.oscillatory
    assert res.p_obs == pytest.approx(2.0, rel=1e-5)
    assert res.q_ext == pytest.approx(q_inf, rel=1e-6)
    assert res.fs == FS_THREE_LEVEL
    assert res.gci_fine < 0.01
    assert res.asymptotic_ratio == pytest.approx(1.0, rel=1e-3)


def test_oscillatory_series_marks_non_monotonic():
    h = [0.01, 0.02, 0.04]
    q = [1.0, 1.2, 1.1]  # ε21>0, ε32<0
    res = gci_triplet(h, q)
    assert not res.ok
    assert res.oscillatory
    assert res.detail == "gci_non_monotonic"
    # Two-level Fs=3 fallback still populates a finite GCI for CSV.
    assert np.isfinite(res.gci_fine)
    assert res.fs == 3.0


def test_unequal_r_study_fracs_monotonic():
    """Study fracs 0.008, 0.012, 0.016 (unequal r) on Q=Qinf+C h^2."""
    fracs = np.array([0.008, 0.012, 0.016])
    chord = 0.241
    h = fracs * chord
    q_inf, c = 5.0, 3.0
    q = q_inf + c * h**2
    res = gci_triplet(h, q)
    assert res.ok
    assert not res.oscillatory
    assert abs(res.r21 - res.r32) > 0.01  # unequal
    assert res.p_obs == pytest.approx(2.0, rel=0.05)
    assert res.q_ext == pytest.approx(q_inf, rel=1e-3)
    assert res.gci_fine < 0.05


def test_two_level_uses_fs3():
    h = [0.01, 0.02]
    q = [1.0, 1.04]
    res = gci_triplet(h, q)
    assert res.ok
    assert res.fs == 3.0
    assert res.detail == "two_level_fs3"
    gci_manual = gci_fine(0.01, 0.02, 1.0, 1.04, p=2.0, fs=3.0)
    assert res.gci_fine == pytest.approx(gci_manual)


def test_gci_fine_formula():
    # r=2, p=2, Fs=1.25, Q1=1, Q2=1.04 → GCI = 1.25*0.04/(4-1)/1 = 0.016666...
    g = gci_fine(1.0, 2.0, 1.0, 1.04, p=2.0, fs=1.25)
    assert g == pytest.approx(1.25 * 0.04 / 3.0)
