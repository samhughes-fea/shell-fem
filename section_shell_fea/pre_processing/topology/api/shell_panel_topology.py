"""Compat shim — import from ``pre_processing.mesh.api.bound_section_shell_inputs``."""

from __future__ import annotations

import warnings

warnings.warn(
    "Import shell_panel_topology from "
    "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.bound_section_shell_inputs.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.bound_section_shell_inputs import (  # noqa: E402,F401
    BoundSectionShellInputs,
    TaperedBoundSectionShellInputs,
    bind_section_shell_inputs,
)

ShellPanelTopology = BoundSectionShellInputs
TaperedShellPanelTopology = TaperedBoundSectionShellInputs
shell_panel_topology_from_shell_inputs = bind_section_shell_inputs

__all__ = [
    "BoundSectionShellInputs",
    "ShellPanelTopology",
    "TaperedBoundSectionShellInputs",
    "TaperedShellPanelTopology",
    "bind_section_shell_inputs",
    "shell_panel_topology_from_shell_inputs",
]
