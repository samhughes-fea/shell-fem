"""Frozen constraint products for station shell FE.

``SectionConstraintSet.equations`` holds MPC relations; ``fixed_dofs`` holds
essential supports. The solver dictionary is produced only by the adapter at
the constrain/reduce boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import math

Constraints = dict[int, list[tuple[int, float]]]


@dataclass(frozen=True, slots=True)
class ConstraintTerm:
    """One master contribution in a linear MPC: ``coeff * q_master``."""

    master_dof: int
    coefficient: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "master_dof", int(self.master_dof))
        coeff = float(self.coefficient)
        if not math.isfinite(coeff):
            raise ValueError("ConstraintTerm.coefficient must be finite")
        object.__setattr__(self, "coefficient", coeff)


@dataclass(frozen=True, slots=True)
class ConstraintEquation:
    """One slave DOF expressed as a linear combination of master DOFs."""

    slave_dof: int
    terms: tuple[ConstraintTerm, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slave_dof", int(self.slave_dof))
        terms = tuple(sorted(self.terms, key=lambda t: t.master_dof))
        masters = [t.master_dof for t in terms]
        if len(set(masters)) != len(masters):
            raise ValueError(
                f"ConstraintEquation slave {self.slave_dof}: duplicate master DOFs"
            )
        if self.slave_dof in masters:
            raise ValueError(
                f"ConstraintEquation slave {self.slave_dof} references itself"
            )
        object.__setattr__(self, "terms", terms)


@dataclass(frozen=True, slots=True)
class SectionConstraintReport:
    """Modes, counts, and ownership tags — not the equation graph."""

    interface_constraint_mode: str
    cap_tie_mode: str
    bc_mode: str
    n_junction_equations: int
    n_cap_tie_equations: int
    n_fixed_dofs: int
    n_cap_links: int
    tie_slave_nodes: frozenset[int]
    junction_slave_dofs: frozenset[int]
    notes: tuple[str, ...] = ()
    cap_tie: object | None = None


@dataclass(frozen=True, slots=True)
class SectionConstraintSet:
    """Immutable composed constraint set for one station solve."""

    equations: tuple[ConstraintEquation, ...]
    fixed_dofs: frozenset[int]
    report: SectionConstraintReport

    def __post_init__(self) -> None:
        equations = tuple(sorted(self.equations, key=lambda eq: eq.slave_dof))
        slaves = [eq.slave_dof for eq in equations]
        if len(set(slaves)) != len(slaves):
            raise ValueError("SectionConstraintSet: duplicate slave DOFs")
        fixed = frozenset(int(d) for d in self.fixed_dofs)
        overlap = set(slaves) & set(fixed)
        if overlap:
            raise ValueError(
                "SectionConstraintSet: slave DOFs also in fixed_dofs: "
                f"{sorted(overlap)[:8]}"
            )
        object.__setattr__(self, "equations", equations)
        object.__setattr__(self, "fixed_dofs", fixed)


def constraint_equations_to_solver_dict(
    equations: Sequence[ConstraintEquation],
) -> Constraints:
    """Adapter immediately before constrain/reduce: typed rows → solver dict."""
    out: Constraints = {}
    for eq in equations:
        out[int(eq.slave_dof)] = [
            (int(t.master_dof), float(t.coefficient)) for t in eq.terms
        ]
    return out


def section_constraint_set_from_solver_inputs(
    constraints: Mapping[int, Sequence[tuple[int, float]]],
    fixed_dofs: Sequence[int] | set[int],
    *,
    interface_constraint_mode: str = "explicit",
    cap_tie_mode: str = "none",
    bc_mode: str = "explicit",
) -> SectionConstraintSet:
    """Wrap a complete explicit solver dict + fixed set (low-level callers)."""
    equations = solver_dict_to_constraint_equations(constraints)
    report = SectionConstraintReport(
        interface_constraint_mode=str(interface_constraint_mode),
        cap_tie_mode=str(cap_tie_mode),
        bc_mode=str(bc_mode),
        n_junction_equations=0,
        n_cap_tie_equations=len(equations),
        n_fixed_dofs=len(set(int(d) for d in fixed_dofs)),
        n_cap_links=0,
        tie_slave_nodes=frozenset(),
        junction_slave_dofs=frozenset(),
    )
    return SectionConstraintSet(
        equations=equations,
        fixed_dofs=frozenset(int(d) for d in fixed_dofs),
        report=report,
    )


def solver_dict_to_constraint_equations(
    constraints: Mapping[int, Sequence[tuple[int, float]]],
) -> tuple[ConstraintEquation, ...]:
    """Normalize an unordered solver dict into sorted immutable equations."""
    equations: list[ConstraintEquation] = []
    for slave in sorted(int(s) for s in constraints):
        acc: dict[int, float] = {}
        for master, coeff in constraints[slave]:
            m = int(master)
            c = float(coeff)
            if not math.isfinite(c) or abs(c) <= 1e-14:
                continue
            acc[m] = acc.get(m, 0.0) + c
        terms = tuple(
            ConstraintTerm(master_dof=m, coefficient=c)
            for m, c in sorted(acc.items())
            if abs(c) > 1e-14
        )
        if terms:
            equations.append(ConstraintEquation(slave_dof=int(slave), terms=terms))
    return tuple(equations)


__all__ = [
    "ConstraintEquation",
    "ConstraintTerm",
    "Constraints",
    "SectionConstraintReport",
    "SectionConstraintSet",
    "constraint_equations_to_solver_dict",
    "section_constraint_set_from_solver_inputs",
    "solver_dict_to_constraint_equations",
]
