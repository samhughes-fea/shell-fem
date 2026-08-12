"""FE re-export lock for section centerline mesh figure helpers."""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea import vis as fe_vis
from blade_structure_utilities.mesh_utilities.section_mesh import visualisation as mesh_vis


def test_fe_vis_reexports_section_mesh_centerline_helpers() -> None:
    assert (
        fe_vis.save_section_centerline_mesh_figure
        is mesh_vis.save_section_centerline_mesh_figure
    )
    assert fe_vis.taxonomy_plot_caption is mesh_vis.taxonomy_plot_caption
