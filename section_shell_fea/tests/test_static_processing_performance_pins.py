"""Exactness and behaviour pins for assembly / reduction / LU / provenance opts."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    AssemblyDiagnostics,
    ShellForceObject,
    ShellFormulationCache,
    ShellStiffnessObject,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.solve.solve_quality import (
    CatastrophicSolveError,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.assembly.assemble_global import (
    assemble_section_system,
    scatter_element_fe,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.contracts.solve_states import (
    ConstrainedShellSystem,
    GlobalShellSystem,
    ReducedShellSystem,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.reduction.reduce_constrained import (
    ReduceConstrainedShellSystem,
    ReductionIdentity,
    ReductionOperator,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.solution.solve_reduced import (
    SolveReducedShellSystem,
    k_reduced_identity,
)


def _tiny_cache() -> tuple[ShellFormulationCache, int]:
    # Two 2-DOF elements sharing a DOF → duplicate force scatter must accumulate.
    ke0 = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=float)
    ke1 = np.array([[3.0, -1.5], [-1.5, 3.0]], dtype=float)
    fe0 = np.array([1.0, 0.5], dtype=float)
    fe1 = np.array([0.25, 0.75], dtype=float)
    stiff = [
        ShellStiffnessObject(
            element_id=0,
            element_type="test",
            K_e=ke0,
            global_dof_indices=np.array([0, 1], dtype=int),
        ),
        ShellStiffnessObject(
            element_id=1,
            element_type="test",
            K_e=ke1,
            global_dof_indices=np.array([1, 2], dtype=int),
        ),
    ]
    force = [
        ShellForceObject(element_id=0, element_type="test", F_e=fe0),
        ShellForceObject(element_id=1, element_type="test", F_e=fe1),
    ]
    cache = ShellFormulationCache(
        stiffness_objects=stiff,
        force_objects=force,
        n_dof_node=1,
    )
    return cache, 3


def test_scatter_element_fe_duplicate_safe():
    f = np.zeros(3)
    scatter_element_fe(np.array([1.0, 2.0]), np.array([0, 0]), f)
    assert f[0] == pytest.approx(3.0)


def test_assemble_section_system_prealloc_parity_and_validate_default():
    cache, n_gdof = _tiny_cache()
    K, f, diag = assemble_section_system(cache, n_gdof, validate=True)
    assert isinstance(diag, AssemblyDiagnostics)
    assert K.shape == (3, 3)
    assert f.shape == (3,)
    # Expected f from manual scatter
    expected_f = np.array([1.0, 0.5 + 0.25, 0.75], dtype=float)
    np.testing.assert_allclose(f, expected_f)
    # Symmetry of assembled K
    assert diag.symmetry_max_abs_diff == pytest.approx(0.0, abs=1e-14)
    K2, f2, diag2 = assemble_section_system(cache, n_gdof, validate=False)
    np.testing.assert_allclose(K.toarray(), K2.toarray())
    np.testing.assert_allclose(f, f2)
    assert diag.nnz == diag2.nnz


def test_reduction_operator_reuse_skips_t_rebuild_and_refreshes_krr():
    n_gdof = 4
    fixed = {0}
    constraints = {3: [(1, 1.0)]}
    constrained = ConstrainedShellSystem(constraints=constraints, fixed=fixed, dof_provenance=[])
    op = ReductionOperator.build(constrained, n_gdof)
    K1 = sp.eye(n_gdof, format="csr") * 2.0
    f1 = np.arange(n_gdof, dtype=float)
    reduced1 = op.apply(GlobalShellSystem(K=K1, f=f1))
    K2 = sp.eye(n_gdof, format="csr") * 5.0
    f2 = np.ones(n_gdof, dtype=float)
    reduced2 = op.apply(GlobalShellSystem(K=K2, f=f2))
    assert reduced1.T is reduced2.T
    assert op.identity == ReductionIdentity.from_constrained(constrained, n_gdof)
    # K_rr must reflect the new global K
    assert float(reduced1.K_reduced[0, 0]) != float(reduced2.K_reduced[0, 0])
    reducer = ReduceConstrainedShellSystem()
    r_a, _ = reducer.run(GlobalShellSystem(K=K1, f=f1), constrained, n_gdof, operator=op)
    r_b, _ = reducer.run(GlobalShellSystem(K=K2, f=f2), constrained, n_gdof, operator=op)
    assert r_a.T is op.T
    assert r_b.T is op.T
    np.testing.assert_allclose(r_a.K_reduced.toarray(), reduced1.K_reduced.toarray())
    np.testing.assert_allclose(r_b.K_reduced.toarray(), reduced2.K_reduced.toarray())


def test_solve_reduced_factor_once_many_rhs_and_stale_guard():
    K = sp.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    f0 = np.array([1.0, 0.0])
    f1 = np.array([0.0, 1.0])
    independent = np.array([0, 1], dtype=int)
    T = sp.eye(2, format="csr")
    reduced = ReducedShellSystem(T=T, K_reduced=K, f_reduced=f0, independent=independent)
    solver = SolveReducedShellSystem(reduced)
    outs = solver.solve_many([f0, f1])
    assert len(outs) == 2
    q0_ref = spla.spsolve(K, f0)
    q1_ref = spla.spsolve(K, f1)
    np.testing.assert_allclose(outs[0].q, q0_ref)
    np.testing.assert_allclose(outs[1].q, q1_ref)
    assert outs[0].factor is outs[1].factor

    # Injected factor with changed K must fail closed under strict identity.
    factor = solver.prepare()
    K_new = sp.csc_matrix(np.array([[10.0, 0.0], [0.0, 10.0]]))
    reduced_new = ReducedShellSystem(T=T, K_reduced=K_new, f_reduced=f0, independent=independent)
    stale = SolveReducedShellSystem(
        reduced_new,
        factor=factor,
        factor_k_identity=k_reduced_identity(K),
        strict_factor_identity=True,
    )
    with pytest.raises(CatastrophicSolveError, match="stale SuperLU"):
        stale.solve(f0)


def test_provenance_lazy_materializes_same_as_eager():
    from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
        SectionLinearStaticRunner,
    )

    n_gdof = 3
    K = sp.eye(n_gdof, format="csr") * 2.0
    f = np.array([1.0, 0.0, 0.0])
    constraints: dict[int, list[tuple[int, float]]] = {}
    fixed = {2}
    eager = SectionLinearStaticRunner.from_global(K, f, constraints, fixed, n_gdof)
    eager.provenance_policy = "eager"
    lazy = SectionLinearStaticRunner.from_global(K, f, constraints, fixed, n_gdof)
    lazy.provenance_policy = "lazy"
    d_e = eager.solve_linear_system_only()
    d_l = lazy.solve_linear_system_only()
    np.testing.assert_allclose(d_e, d_l)
    assert lazy.index_maps.constraint_map == []
    maps = lazy.ensure_provenance_maps()
    assert len(maps.constraint_map) == n_gdof
    assert len(eager.index_maps.constraint_map) == n_gdof
    assert [e.global_dof for e in maps.constraint_map] == [
        e.global_dof for e in eager.index_maps.constraint_map
    ]
