"""Compat shim — import from ``pre_processing.mesh.section_shell_assembly_map``."""

from __future__ import annotations

import warnings

warnings.warn(
    "Import connected_panel_topology from "
    "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.section_shell_assembly_map.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.section_shell_assembly_map import (  # noqa: E402,F401
    SectionShellAssemblyMap,
    build_section_shell_assembly_map,
)

ConnectedPanelTopology = SectionShellAssemblyMap
build_connected_panel_maps = build_section_shell_assembly_map

__all__ = [
    "ConnectedPanelTopology",
    "SectionShellAssemblyMap",
    "build_connected_panel_maps",
    "build_section_shell_assembly_map",
]
