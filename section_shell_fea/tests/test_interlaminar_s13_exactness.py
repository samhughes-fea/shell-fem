"""FEA-FI-002: interlaminar allowable must not silently substitute S12 for S13."""

from __future__ import annotations

from types import SimpleNamespace

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_quaternary.compute_hashin_fi import (
    interlaminar_allowable_from_lam,
    interlaminar_allowable_pa,
)


def test_interlaminar_allowable_prefers_s13():
    ply = SimpleNamespace(S13=40e6, S12=60e6)
    assert interlaminar_allowable_pa(ply) == 40e6


def test_interlaminar_allowable_refuses_silent_s12():
    ply = SimpleNamespace(S13=0.0, S12=60e6)
    assert interlaminar_allowable_pa(ply) == 0.0
    assert interlaminar_allowable_pa(ply, allow_s12_fallback=True) == 60e6


def test_interlaminar_allowable_from_lam_first_ply():
    ply0 = SimpleNamespace(S13=0.0, S12=50e6)
    ply1 = SimpleNamespace(S13=80e6, S12=10e6)
    lam = SimpleNamespace(plies=[(ply0, 0.0), (ply1, 45.0)])
    assert interlaminar_allowable_from_lam(lam) == 0.0
    assert interlaminar_allowable_from_lam(lam, allow_s12_fallback=True) == 50e6
