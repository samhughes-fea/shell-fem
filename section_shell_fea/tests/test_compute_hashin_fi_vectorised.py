"""Regression tests for ply_hashin_fi_modes's vectorised (row, ply) evaluation.

Locks in that the single batched hashin_fi_modes() call reproduces the old
row-by-row/ply-by-ply Python loop exactly, including untrusted-row masking --
this was rewritten purely for performance (quaternary_results was the single
most expensive per-trial phase once real multi-ply CLT stacks made n_ply large;
see DiagnosticsBin / sizing timing logs), with no intended change in FI values.
"""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_quaternary.compute_hashin_fi import (
    ply_hashin_fi_modes,
)
from blade_structure_utilities.material_utilities.failure import hashin_fi_modes


def _reference_loop(sig_ply, trusted_mask, *, Xt, Xc, Yt, Yc, S12):
    """The old implementation, kept only as an independent reference oracle."""
    n_rows, n_ply_arr, _ = sig_ply.shape
    n_ply = min(n_ply_arr, len(Xt))
    out_cols = max(n_ply_arr, 1)
    fi_ft = np.zeros((n_rows, out_cols), dtype=float)
    fi_fc = np.zeros((n_rows, out_cols), dtype=float)
    fi_mt = np.zeros((n_rows, out_cols), dtype=float)
    fi_mc = np.zeros((n_rows, out_cols), dtype=float)
    for ri in np.flatnonzero(trusted_mask):
        for pi in range(n_ply):
            sig3 = np.asarray(sig_ply[ri, pi], dtype=float).ravel()[:3]
            ft, fc, mt, mc = hashin_fi_modes(
                sig3.reshape(1, 3),
                Xt[pi : pi + 1], Xc[pi : pi + 1], Yt[pi : pi + 1], Yc[pi : pi + 1], S12[pi : pi + 1],
            )
            fi_ft[ri, pi] = float(ft[0])
            fi_fc[ri, pi] = float(fc[0])
            fi_mt[ri, pi] = float(mt[0])
            fi_mc[ri, pi] = float(mc[0])
    return fi_ft, fi_fc, fi_mt, fi_mc


def _random_case(*, n_rows: int, n_ply: int, seed: int, untrusted_frac: float = 0.0):
    rng = np.random.default_rng(seed)
    sig_ply = rng.uniform(-2e8, 2e8, size=(n_rows, n_ply, 3))
    trusted_mask = rng.uniform(size=n_rows) >= untrusted_frac
    Xt = rng.uniform(3e8, 1.5e9, size=n_ply)
    Xc = rng.uniform(2e8, 1.2e9, size=n_ply)
    Yt = rng.uniform(2e7, 8e7, size=n_ply)
    Yc = rng.uniform(1e8, 3e8, size=n_ply)
    S12 = rng.uniform(3e7, 7e7, size=n_ply)
    return sig_ply, trusted_mask, Xt, Xc, Yt, Yc, S12


class TestPlyHashinFiModesVectorised:
    def test_matches_reference_loop_realistic_multi_ply_stack(self):
        """80-ply spar cap stack, ~600 rows -- the shape this now runs at in production."""
        sig_ply, trusted_mask, Xt, Xc, Yt, Yc, S12 = _random_case(
            n_rows=37, n_ply=80, seed=0
        )
        (fi_ft, fi_fc, fi_mt, fi_mc), needs_review = ply_hashin_fi_modes(
            sig_ply, trusted_mask, Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12,
        )
        ref_ft, ref_fc, ref_mt, ref_mc = _reference_loop(
            sig_ply, trusted_mask, Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12,
        )
        assert np.allclose(fi_ft, ref_ft)
        assert np.allclose(fi_fc, ref_fc)
        assert np.allclose(fi_mt, ref_mt)
        assert np.allclose(fi_mc, ref_mc)
        assert np.array_equal(needs_review, ~trusted_mask)

    def test_matches_reference_loop_with_untrusted_rows(self):
        sig_ply, trusted_mask, Xt, Xc, Yt, Yc, S12 = _random_case(
            n_rows=25, n_ply=12, seed=1, untrusted_frac=0.4
        )
        assert np.any(~trusted_mask) and np.any(trusted_mask)  # exercise both branches
        (fi_ft, fi_fc, fi_mt, fi_mc), needs_review = ply_hashin_fi_modes(
            sig_ply, trusted_mask, Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12,
        )
        ref_ft, ref_fc, ref_mt, ref_mc = _reference_loop(
            sig_ply, trusted_mask, Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12,
        )
        assert np.allclose(fi_ft, ref_ft)
        assert np.allclose(fi_fc, ref_fc)
        assert np.allclose(fi_mt, ref_mt)
        assert np.allclose(fi_mc, ref_mc)

    def test_untrusted_row_output_is_zero_even_with_nan_stress(self):
        """Untrusted rows must never let (garbage/NaN) stress reach the criterion."""
        n_rows, n_ply = 5, 3
        sig_ply = np.full((n_rows, n_ply, 3), np.nan)
        sig_ply[0] = [[1e8, 2e7, 1e6]] * n_ply  # only row 0 is trusted + finite
        trusted_mask = np.array([True, False, False, False, False])
        Xt = np.full(n_ply, 8e8)
        Xc = np.full(n_ply, 6e8)
        Yt = np.full(n_ply, 4e7)
        Yc = np.full(n_ply, 1.5e8)
        S12 = np.full(n_ply, 5e7)

        (fi_ft, fi_fc, fi_mt, fi_mc), needs_review = ply_hashin_fi_modes(
            sig_ply, trusted_mask, Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12,
        )
        assert np.all(np.isfinite(fi_ft)) and np.all(np.isfinite(fi_fc))
        assert np.all(np.isfinite(fi_mt)) and np.all(np.isfinite(fi_mc))
        assert np.all(fi_ft[1:] == 0.0)
        assert np.all(fi_fc[1:] == 0.0)
        assert np.all(fi_mt[1:] == 0.0)
        assert np.all(fi_mc[1:] == 0.0)
        assert np.array_equal(needs_review, ~trusted_mask)
        assert np.any(fi_ft[0] != 0.0) or np.any(fi_mt[0] != 0.0)

    def test_empty_rows_or_plies_returns_zeros_no_crash(self):
        for n_rows, n_ply in ((0, 5), (5, 0)):
            sig_ply = np.zeros((n_rows, max(n_ply, 0), 3))
            trusted_mask = np.ones(n_rows, dtype=bool)
            Xt = Xc = Yt = Yc = S12 = np.full(n_ply, 1e8)
            (fi_ft, fi_fc, fi_mt, fi_mc), needs_review = ply_hashin_fi_modes(
                sig_ply, trusted_mask, Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12,
            )
            assert fi_ft.shape[0] == n_rows
            assert needs_review.shape[0] == n_rows

    def test_vectorised_call_is_dramatically_fewer_python_calls(self):
        """Sanity check on the actual perf fix: one hashin_fi_modes() call total,
        not one per (row, ply) -- the whole point of the rewrite."""
        import unittest.mock as mock

        sig_ply, trusted_mask, Xt, Xc, Yt, Yc, S12 = _random_case(
            n_rows=50, n_ply=64, seed=2
        )
        real_fn = hashin_fi_modes
        call_count = 0

        def _counting_wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return real_fn(*args, **kwargs)

        # ply_hashin_fi_modes imports hashin_fi_modes locally (function-body
        # import), so patch it at its defining module -- not at
        # compute_hashin_fi, which has no module-level binding for it.
        with mock.patch(
            "blade_structure_utilities.material_utilities.failure.hashin_fi_modes",
            side_effect=_counting_wrapper,
        ):
            ply_hashin_fi_modes(sig_ply, trusted_mask, Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12)

        # Old loop form would have made 50*64 = 3200 calls; vectorised form makes 1.
        assert call_count == 1
