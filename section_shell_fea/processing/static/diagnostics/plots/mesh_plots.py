"""Compatibility re-export — centerline mesh figures live in ``section_mesh.visualisation``."""

from __future__ import annotations

from blade_structure_utilities.mesh_utilities.section_mesh.visualisation.centerline_mesh import (  # noqa: F401
    draw_centerline_mesh_on_ax,
    draw_midline_strips_on_ax,
    draw_panel_patches_on_ax,
    save_section_centerline_mesh_figure,
    taxonomy_plot_caption,
)

__all__ = [
    "draw_centerline_mesh_on_ax",
    "draw_midline_strips_on_ax",
    "draw_panel_patches_on_ax",
    "save_section_centerline_mesh_figure",
    "taxonomy_plot_caption",
]
