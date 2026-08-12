"""Reduced-solve quality metrics and DiagnosticsBin cascade trust."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.solve_quality import (
    CatastrophicSolveError,
    assess_solve_quality,
    residual_rel_reduced,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.solve_reduced import (
    SolveReducedShellSystem,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.solve_states import (
    ReducedShellSystem,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cascade_mesh_convergence import (
    run_cascade_at_frac,
)


def test_residual_rel_identity():
    k = sp.eye(3, format="csc")
    q = np.array([1.0, 2.0, 3.0])
    f = q.copy()
    assert residual_rel_reduced(k, q, f) < 1e-14


def test_assess_solve_quality_soft_fail_on_large_residual():
    report = assess_solve_quality(residual_rel=1e-3, pivot_ratio=10.0)
    assert report.ok is False
    assert report.catastrophic is False


def test_assess_solve_quality_catastrophic_non_finite_q():
    report = assess_solve_quality(
        residual_rel=0.0,
        pivot_ratio=1.0,
        q=np.array([1.0, np.nan]),
    )
    assert report.catastrophic is True


def test_solve_reduced_well_posed():
    k = sp.csc_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    f = np.array([1.0, 0.0])
    reduced = ReducedShellSystem(
        T=sp.eye(2, format="csr"),
        K_reduced=k,
        f_reduced=f,
        independent=np.array([0, 1], dtype=np.int64),
    )
    solved = SolveReducedShellSystem(reduced).solve()
    assert solved.q.shape == (2,)
    assert np.all(np.isfinite(solved.q))


def test_healthy_cascade_sets_cascade_trust_ok():
    row = run_cascade_at_frac(0.024, cap_tie_mode="skin_offset")
    assert row.cascade_trust_ok is True
    assert row.mesh_ok is True
    assert row.solve_residual_rel is not None
    assert float(row.solve_residual_rel) < 1e-8
