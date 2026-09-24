"""Compatibility shim — cap-tie geometry and MPCs live in ``constraints.cap_tie``."""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.cap_tie.geometry import (
    CAP_TIE_MODES,
    CapTieLink,
    CapTieReport,
    build_cap_skin_ties,
    kind_of as _kind_of,
    project_onto_polyline as _project_onto_polyline,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.cap_tie.mpcs import (
    build_cap_tie_constraints,
    cap_tie_constraints,
)

__all__ = [
    "CAP_TIE_MODES",
    "CapTieLink",
    "CapTieReport",
    "build_cap_skin_ties",
    "build_cap_tie_constraints",
    "cap_tie_constraints",
    "_kind_of",
    "_project_onto_polyline",
]
