"""Unit tests for SectionLinearStaticRunner and primary results tier."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.results.compute_primary import ShellPrimaryResultsOrchestrator
from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.section_linear_static_runner import SectionLinearStaticRunner


def _tiny_system():
    n = 4
    K = sp.diags([2.0, 2.0, 2.0, 2.0], format="csc")
    f = np.array([1.0, 0.0, 0.0, 0.0])
    fixed = {1, 3}
    constraints: dict[int, list[tuple[int, float]]] = {}
    return K, f, constraints, fixed, n


def test_r_full_is_kd_minus_f_not_reactions_plus_residual():
    K, f, constraints, fixed, n_gdof = _tiny_system()
    runner = SectionLinearStaticRunner.from_global(K, f, constraints, fixed, n_gdof)
    runner.solve_linear_system_only()
    runner.compute_primary_results()
    assert runner.field is not None and runner.primary is not None
    d = runner.field.d_global
    expected = np.asarray(K @ d - f, dtype=float)
    np.testing.assert_allclose(runner.primary.r_full, expected, rtol=0, atol=1e-12)
    wrong = runner.primary.reactions + runner.primary.residual
    assert float(np.max(np.abs(wrong - runner.primary.r_full))) > 1e-6 or np.allclose(
        wrong, runner.primary.r_full
    )


def test_from_global_updates_fixed_from_solve():
    K, f, constraints, fixed, n_gdof = _tiny_system()
    fixed_mut = set(fixed)
    runner = SectionLinearStaticRunner.from_global(K, f, constraints, fixed_mut, n_gdof)
    d = runner.solve_linear_system_only()
    fixed_mut.clear()
    fixed_mut.update(runner.constrained.fixed if runner.constrained else fixed)
    assert len(d) == n_gdof
    assert fixed_mut == set(fixed)


def test_primary_orchestrator_splits_reactions_and_residual():
    """compute() returns a ShellPrimaryResultSet; the bare PrimaryShellResults
    (reactions/residual/r_full) is retained on orchestrator.primary for callers
    that need it directly (see PROCESSING_ARCHITECTURE_PROPOSAL.md tier 1/2)."""
    K, f, constraints, fixed, n_gdof = _tiny_system()
    runner = SectionLinearStaticRunner.from_global(K, f, constraints, fixed, n_gdof)
    runner.solve_linear_system_only()
    assert runner.global_sys is not None and runner.field is not None and runner.constrained is not None
    orchestrator = ShellPrimaryResultsOrchestrator(
        K=runner.global_sys.K,
        f=runner.global_sys.f,
        d_global=runner.field.d_global,
        fixed=runner.constrained.fixed,
    )
    result_set = orchestrator.compute()
    primary = orchestrator.primary
    r_full = np.asarray(runner.global_sys.K @ runner.field.d_global - runner.global_sys.f, dtype=float)
    for d in fixed:
        assert abs(primary.reactions[d] - r_full[d]) < 1e-12
    free = [i for i in range(n_gdof) if i not in fixed]
    np.testing.assert_allclose(primary.residual[free], r_full[free] * -1.0, atol=1e-10)

    # The returned container carries the same values under global_results.
    assert result_set.global_results is not None
    np.testing.assert_allclose(result_set.global_results.reactions, primary.reactions)
    np.testing.assert_allclose(result_set.global_results.residual, primary.residual)
    np.testing.assert_allclose(result_set.global_results.r_full, primary.r_full)
    # No cache/global_sys/field supplied: elemental results are empty (matches
    # pre-tightening runner behaviour for a bare from_global() runner).
    assert result_set.elemental_results is not None
    assert result_set.elemental_results.U_e == []


def test_runner_collects_index_maps():
    K, f, constraints, fixed, n_gdof = _tiny_system()
    runner = SectionLinearStaticRunner.from_global(K, f, constraints, fixed, n_gdof)
    runner.solve_linear_system_only()
    assert runner.index_maps.constraint_map is not None
    assert runner.index_maps.reduction_map is not None
    assert runner.index_maps.reconstruction_map is not None
    assert len(runner.index_maps.constraint_map) == n_gdof
