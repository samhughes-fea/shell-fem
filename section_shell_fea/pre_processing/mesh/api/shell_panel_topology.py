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


def build_shell_panel_topology(*args, **kwargs):
    warnings.warn(
        "build_shell_panel_topology is deprecated; use "
        "tests._panel_topology_factory.build_demo_bound_section_shell_inputs.",
        DeprecationWarning,
        stacklevel=2,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.visualisation.demos.demo_bound_section_shell_inputs import (
        build_demo_bound_section_shell_inputs,
    )

    return build_demo_bound_section_shell_inputs(*args, **kwargs)


def build_tapered_shell_panel_topology(*args, **kwargs):
    warnings.warn(
        "build_tapered_shell_panel_topology is deprecated; use "
        "tests._panel_topology_factory.build_demo_tapered_bound_section_shell_inputs.",
        DeprecationWarning,
        stacklevel=2,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.visualisation.demos.demo_bound_section_shell_inputs import (
        build_demo_tapered_bound_section_shell_inputs,
    )

    return build_demo_tapered_bound_section_shell_inputs(*args, **kwargs)


__all__ = [
    "BoundSectionShellInputs",
    "ShellPanelTopology",
    "TaperedBoundSectionShellInputs",
    "TaperedShellPanelTopology",
    "bind_section_shell_inputs",
    "build_shell_panel_topology",
    "build_tapered_shell_panel_topology",
    "shell_panel_topology_from_shell_inputs",
]
