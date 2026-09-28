"""Compat shim — import from ``pre_processing.topology.connected_panel_topology``."""

from __future__ import annotations

import warnings

warnings.warn(
    "Import connected_panel_topology from "
    "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.connected_panel_topology.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.connected_panel_topology import (  # noqa: E402,F401
    ConnectedPanelTopology,
    build_connected_panel_maps,
)

__all__ = ["ConnectedPanelTopology", "build_connected_panel_maps"]
