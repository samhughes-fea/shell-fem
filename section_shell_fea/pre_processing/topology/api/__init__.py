"""Compat shim — topology.api re-exports mesh.api."""

from __future__ import annotations

import warnings

warnings.warn(
    "Import topology API from "
    "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api import (  # noqa: E402,F401
    BoundSectionShellInputs,
    TaperedBoundSectionShellInputs,
    bind_section_shell_inputs,
)

__all__ = [
    "BoundSectionShellInputs",
    "TaperedBoundSectionShellInputs",
    "bind_section_shell_inputs",
]
