"""MITC4 element-specific helpers.

ABD collection lives in ``pre_processing.constitutive.panel_abd``.
Strip-mesh symbols live in ``mesh_utilities.section_mesh`` (import there).
"""

from ....constitutive.panel_abd import collect_panel_abd_stack

__all__ = [
    "collect_panel_abd_stack",
]
