"""WP0/WP1 constraint product validation and solver-dict adapter."""

from __future__ import annotations

import math

import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.types import (
    ConstraintEquation,
    ConstraintTerm,
    SectionConstraintReport,
    SectionConstraintSet,
    constraint_equations_to_solver_dict,
    section_constraint_set_from_solver_inputs,
    solver_dict_to_constraint_equations,
)


def test_constraint_term_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        ConstraintTerm(0, math.nan)


def test_equation_rejects_self_reference_and_duplicate_masters():
    with pytest.raises(ValueError, match="itself"):
        ConstraintEquation(1, (ConstraintTerm(1, 1.0),))
    with pytest.raises(ValueError, match="duplicate"):
        ConstraintEquation(1, (ConstraintTerm(2, 1.0), ConstraintTerm(2, 0.5)))


def test_set_rejects_fixed_slave_overlap():
    eq = ConstraintEquation(3, (ConstraintTerm(0, 1.0),))
    report = SectionConstraintReport(
        interface_constraint_mode="shared",
        cap_tie_mode="none",
        bc_mode="contour_support",
        n_junction_equations=0,
        n_cap_tie_equations=1,
        n_fixed_dofs=1,
        n_cap_links=0,
        tie_slave_nodes=frozenset(),
        junction_slave_dofs=frozenset(),
    )
    with pytest.raises(ValueError, match="fixed_dofs"):
        SectionConstraintSet(equations=(eq,), fixed_dofs=frozenset({3}), report=report)


def test_set_rejects_duplicate_slave_equations():
    eq_a = ConstraintEquation(3, (ConstraintTerm(0, 1.0),))
    eq_b = ConstraintEquation(3, (ConstraintTerm(1, 1.0),))
    report = SectionConstraintReport(
        interface_constraint_mode="shared",
        cap_tie_mode="none",
        bc_mode="contour_support",
        n_junction_equations=0,
        n_cap_tie_equations=2,
        n_fixed_dofs=0,
        n_cap_links=0,
        tie_slave_nodes=frozenset(),
        junction_slave_dofs=frozenset(),
    )
    with pytest.raises(ValueError, match="duplicate slave"):
        SectionConstraintSet(equations=(eq_a, eq_b), fixed_dofs=frozenset(), report=report)


def test_solver_dict_round_trip_is_sorted_and_coalesced():
    raw = {
        5: [(2, 0.5), (3, 0.25), (2, 0.5)],
        1: [(0, 1.0)],
    }
    eqs = solver_dict_to_constraint_equations(raw)
    assert [eq.slave_dof for eq in eqs] == [1, 5]
    assert eqs[1].terms[0].master_dof == 2
    assert eqs[1].terms[0].coefficient == pytest.approx(1.0)
    back = constraint_equations_to_solver_dict(eqs)
    assert list(back) == [1, 5]


def test_cyclic_cap_tie_master_expansion_is_rejected():
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.cap_tie.mpcs import (
        _expand_master_terms,
    )

    cyclic = {1: [(2, 1.0)], 2: [(1, 1.0)]}
    with pytest.raises(ValueError, match="cyclic"):
        _expand_master_terms([(1, 1.0)], cyclic)


def test_explicit_low_level_wrapper_never_infers_station_policy():
    cs = section_constraint_set_from_solver_inputs({4: [(0, 1.0)]}, {0, 1, 2})
    assert cs.report.interface_constraint_mode == "explicit"
    assert cs.fixed_dofs == frozenset({0, 1, 2})
    assert len(cs.equations) == 1
