"""Compatibility shim — FE thin-wall reference/Vlasov adapters moved.

Canonical home:
``blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.reference_thin_wall``.

This package is a time-bounded re-export (see the deprecation register entry
``compat-fe-loading-thin-wall-shim``). Import from ``reference_thin_wall``
instead; the canonical thin-wall *theory* lives in
``geometry_utilities.section_thin_wall_model``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Import FE thin-wall reference/Vlasov adapters from "
    "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing."
    "reference_thin_wall. loading.thin_wall is a temporary re-export.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.reference_thin_wall import (  # noqa: E402,F401
    SectionVlasovResult,
    compute_section_vlasov,
    omega_vertices_open_chain,
    open_outline_from_airfoil,
    shear_center_outer_skin_loop,
)

__all__ = [
    "SectionVlasovResult",
    "compute_section_vlasov",
    "omega_vertices_open_chain",
    "open_outline_from_airfoil",
    "shear_center_outer_skin_loop",
]
