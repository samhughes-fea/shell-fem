"""Thin re-export — canonical home is pre_processing.mesh.connected_panel_topology."""

from blade_structure_utilities.section_shell_model.pre_processing.mesh.connected_panel_topology import (  # noqa: F401
    ConnectedPanelTopology,
    build_connected_panel_maps,
)

__all__ = ["ConnectedPanelTopology", "build_connected_panel_maps"]
