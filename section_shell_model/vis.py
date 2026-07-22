"""Canonical visualisation entrypoint for section shell model figures."""

from __future__ import annotations

from .processing.static.diagnostics.mesh_plots import (  # noqa: F401
    save_section_centerline_mesh_figure,
    taxonomy_plot_caption,
)
from .processing.static.diagnostics.example_plots import (  # noqa: F401
    save_clpt_fi_on_section_geometry,
    save_clpt_ply_figure,
    save_loads_provenance_png,
    save_mitc4_all_panels_resultants_montage,
    save_mitc4_fi_figure,
    save_mitc4_resultants_figure,
    save_mitc4_section_mesh_figure,
    save_mitc4_v2_dashboard_figure,
    save_mitc4_v2_section_mesh_figure,
    save_multi_panel_hashin_montage,
    save_panel_along_contour_figure,
    save_panel_buckling_figure,
    save_panel_stiffness_properties_figure,
    save_ply_stress_breakdown_figure,
    save_reserve_factor_on_section_geometry,
    save_shell_mesh_figure,
    save_thin_wall_stress_figures,
    save_vlasov_all_panels_figure,
    save_warping_stress_on_section_geometry,
)

save_shell_dashboard_figure = save_mitc4_v2_dashboard_figure
save_shell_section_mesh_figure = save_mitc4_v2_section_mesh_figure

__all__ = [
    "save_section_centerline_mesh_figure",
    "taxonomy_plot_caption",
    "save_clpt_fi_on_section_geometry",
    "save_clpt_ply_figure",
    "save_loads_provenance_png",
    "save_mitc4_all_panels_resultants_montage",
    "save_mitc4_fi_figure",
    "save_mitc4_resultants_figure",
    "save_mitc4_section_mesh_figure",
    "save_mitc4_v2_dashboard_figure",
    "save_mitc4_v2_section_mesh_figure",
    "save_multi_panel_hashin_montage",
    "save_panel_along_contour_figure",
    "save_panel_buckling_figure",
    "save_panel_stiffness_properties_figure",
    "save_ply_stress_breakdown_figure",
    "save_reserve_factor_on_section_geometry",
    "save_shell_mesh_figure",
    "save_thin_wall_stress_figures",
    "save_shell_dashboard_figure",
    "save_shell_section_mesh_figure",
    "save_vlasov_all_panels_figure",
    "save_warping_stress_on_section_geometry",
]
