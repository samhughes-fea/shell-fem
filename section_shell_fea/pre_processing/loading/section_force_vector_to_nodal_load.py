"""Compat shim — import from ``processing.static.operations.assembly.load_application``."""

from __future__ import annotations

import warnings

warnings.warn(
    "Import section_force_vector_to_nodal_load from "
    "blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.assembly.load_application.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.assembly.load_application import (  # noqa: E402,F401
    section_force_vector_to_nodal_load,
)

__all__ = ["section_force_vector_to_nodal_load"]
