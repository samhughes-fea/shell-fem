"""Station constraint subsystem.

Owns junction MPCs, cap ties, essential supports, composition, and constraint
provenance. Cap ties are rigid-offset MPCs, not elements.
"""

from __future__ import annotations

from .cap_tie.geometry import CAP_TIE_MODES, CapTieLink, CapTieReport, build_cap_skin_ties
from .cap_tie.mpcs import build_cap_tie_constraints, cap_tie_constraints
from .compose import build_section_constraint_set
from .junction import (
    build_clamp_supports,
    build_cluster_basis_constraints,
    build_cluster_basis_supports,
    build_hybrid_collinear_supports,
)
from .types import (
    ConstraintEquation,
    ConstraintTerm,
    Constraints,
    SectionConstraintReport,
    SectionConstraintSet,
    constraint_equations_to_solver_dict,
    section_constraint_set_from_solver_inputs,
    solver_dict_to_constraint_equations,
)

__all__ = [
    "CAP_TIE_MODES",
    "CapTieLink",
    "CapTieReport",
    "ConstraintEquation",
    "ConstraintTerm",
    "Constraints",
    "SectionConstraintReport",
    "SectionConstraintSet",
    "build_cap_skin_ties",
    "build_cap_tie_constraints",
    "build_clamp_supports",
    "build_cluster_basis_constraints",
    "build_cluster_basis_supports",
    "build_hybrid_collinear_supports",
    "build_section_constraint_set",
    "cap_tie_constraints",
    "constraint_equations_to_solver_dict",
    "section_constraint_set_from_solver_inputs",
    "solver_dict_to_constraint_equations",
]
