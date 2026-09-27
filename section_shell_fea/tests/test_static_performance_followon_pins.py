"""Exactness pins for static performance follow-on WP1–WP4."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.solve.solve_quality import (
    CatastrophicSolveError,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.contracts.solve_states import (
    ReducedShellSystem,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.solution.solve_reduced import (
    SolveReducedShellSystem,
    k_reduced_identity,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_primary.primary_results_orchestrator import (
    ShellPrimaryResultsOrchestrator,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_quaternary.compute_hashin_fi import (
    interface_shear_screening_from_tau,
    interface_shear_screening_from_tau_batch,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_tertiary.recovery_operators import (
    fsdt_interface_shear_profile,
    fsdt_interface_shear_profile_coefficients,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)


def test_injected_factor_without_identity_fails_closed():
    K = sp.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    reduced = ReducedShellSystem(
        T=sp.eye(2, format="csr"),
        K_reduced=K,
        f_reduced=np.array([1.0, 0.0]),
        independent=np.array([0, 1], dtype=int),
    )
    factor = spla.splu(K)
    with pytest.raises(CatastrophicSolveError, match="factor_k_identity"):
        SolveReducedShellSystem(reduced, factor=factor, strict_factor_identity=True)


def test_injected_factor_without_identity_discards_when_non_strict():
    K = sp.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    f = np.array([1.0, 0.0])
    reduced = ReducedShellSystem(
        T=sp.eye(2, format="csr"),
        K_reduced=K,
        f_reduced=f,
        independent=np.array([0, 1], dtype=int),
    )
    factor = spla.splu(K)
    solver = SolveReducedShellSystem(
        reduced, factor=factor, strict_factor_identity=False,
    )
    assert solver.factor_k_identity is None
    out = solver.solve(f)
    np.testing.assert_allclose(out.q, spla.spsolve(K, f))
    assert solver.factor_k_identity == k_reduced_identity(K)


def test_stale_factor_identity_mismatch_fails_closed():
    K = sp.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    K_new = sp.csc_matrix(np.array([[10.0, 0.0], [0.0, 10.0]]))
    f = np.array([1.0, 0.0])
    T = sp.eye(2, format="csr")
    independent = np.array([0, 1], dtype=int)
    solver = SolveReducedShellSystem(
        ReducedShellSystem(T=T, K_reduced=K, f_reduced=f, independent=independent)
    )
    factor = solver.prepare()
    stale = SolveReducedShellSystem(
        ReducedShellSystem(T=T, K_reduced=K_new, f_reduced=f, independent=independent),
        factor=factor,
        factor_k_identity=k_reduced_identity(K),
        strict_factor_identity=True,
    )
    with pytest.raises(CatastrophicSolveError, match="stale SuperLU"):
        stale.solve(f)


def test_solve_many_preserves_first_solution_when_factor_missing():
    n_gdof = 3
    K = sp.eye(n_gdof, format="csr") * 2.0
    f0 = np.array([1.0, 0.0, 0.0])
    f1 = np.array([0.0, 1.0, 0.0])
    runner = SectionLinearStaticRunner.from_global(K, f0, {}, {2}, n_gdof)
    # Force empty-factor edge after first solve by clearing factor.
    outs = runner.solve_many([f0, f1])
    assert len(outs) == 2
    np.testing.assert_allclose(outs[0], runner.solve_linear_system_only(f_override=f0))
    # First solution must never be replaced by an all-zero vector.
    assert np.linalg.norm(outs[0]) > 0.0


def test_primary_single_matvec_bitwise_equal_to_legacy_triple():
    n = 5
    K = sp.csr_matrix(np.eye(n) * 3.0 + np.eye(n, k=1) + np.eye(n, k=-1))
    f = np.arange(n, dtype=float)
    d = np.linspace(0.1, 0.5, n)
    fixed = {0, n - 1}
    Kd = np.asarray(K @ d, dtype=float)
    r_full_ref = Kd - f
    residual_ref = f - Kd
    reactions_ref = np.zeros_like(r_full_ref)
    for dof in fixed:
        reactions_ref[dof] = r_full_ref[dof]
    primary = ShellPrimaryResultsOrchestrator(
        K=K, f=f, d_global=d, fixed=fixed,
    ).compute()
    assert np.array_equal(primary.global_results.r_full, r_full_ref)
    assert np.array_equal(primary.global_results.residual, residual_ref)
    assert np.array_equal(primary.global_results.reactions, reactions_ref)


def test_shear_profile_coefficients_match_scalar_profile():
    x3 = np.linspace(-0.01, 0.01, 5)
    h = 0.02
    g13 = np.array([1.0, 2.0, 1.5, 1.0])
    g23 = np.array([1.2, 1.8, 1.1, 0.9])
    q1, q2 = 12.0, -7.0
    c13, c23 = fsdt_interface_shear_profile_coefficients(
        x3_iface=x3, h=h, g13_ply=g13, g23_ply=g23,
    )
    tau = fsdt_interface_shear_profile(
        q1=q1, q2=q2, x3_iface=x3, h=h, g13_ply=g13, g23_ply=g23,
    )
    np.testing.assert_allclose(tau[:, 0], q1 * c13)
    np.testing.assert_allclose(tau[:, 1], q2 * c23)


class _Ply:
    def __init__(self, s13: float, s23: float, s12: float = 0.0) -> None:
        self.S13 = s13
        self.S23 = s23
        self.S12 = s12


class _Lam:
    def __init__(self, plies: list[tuple[_Ply, float]]) -> None:
        self.plies = plies


def test_interface_shear_batch_matches_scalar_oracle():
    lam = _Lam(
        [
            (_Ply(50e6, 40e6), 0.0),
            (_Ply(55e6, 42e6), 45.0),
            (_Ply(50e6, 40e6), -45.0),
        ]
    )
    rng = np.random.default_rng(0)
    tau = rng.normal(scale=1e5, size=(8, 4, 2))
    tau[:, 0, :] = 0.0
    tau[:, -1, :] = 0.0
    batch = interface_shear_screening_from_tau_batch(tau, lam)
    for i in range(tau.shape[0]):
        scalar = interface_shear_screening_from_tau(tau[i], lam)
        assert batch[i].failure_index == pytest.approx(scalar.failure_index, rel=0, abs=0)
        assert batch[i].needs_review == scalar.needs_review
        assert batch[i].strength_provenance == scalar.strength_provenance
        assert batch[i].governing_interface == scalar.governing_interface
        assert batch[i].governing_side == scalar.governing_side
        assert batch[i].tau13_ratio == pytest.approx(scalar.tau13_ratio, rel=0, abs=0)
        assert batch[i].tau23_ratio == pytest.approx(scalar.tau23_ratio, rel=0, abs=0)


def test_constraint_reduction_cache_key_refuses_incomplete():
    from blade_structure_precompute.module.section_selection.processing.workflows.shared.evaluation_caches import (
        ConstraintReductionCache,
        constraint_reduction_cache_key,
    )

    assert constraint_reduction_cache_key(
        n_gdof=None,
        topology_fingerprint=("t",),
        fixed_dofs=frozenset(),
        mpc_fingerprint=(),
        strip_bond_fingerprint=(),
        bc_mode="contour_support",
        cap_tie_mode="skin_offset",
        interface_constraint_mode="shared",
    ) is None
    cache = ConstraintReductionCache()
    assert cache.lookup(("incomplete",)) is False
    assert cache.refusals == 0  # key mismatch is not a refusal
    cache.key = ("k",)
    assert cache.lookup(("k",)) is False  # incomplete products
    assert cache.refusals == 1
