"""FE thin-wall sectorial-warping and closed-cell Vlasov reference adapters.

These are the FE preprocessing *reference/Vlasov adapters*. The canonical
thin-wall **theory** lives in
``geometry_utilities.section_thin_wall_model``; the adapters here wrap that
theory for the section-shell FE worker (pre-solve).
"""

from __future__ import annotations

from .section_vlasov import SectionVlasovResult, compute_section_vlasov
from .sectorial_warping import (
    omega_vertices_open_chain,
    open_outline_from_airfoil,
    shear_center_outer_skin_loop,
)

__all__ = [
    "SectionVlasovResult",
    "compute_section_vlasov",
    "omega_vertices_open_chain",
    "open_outline_from_airfoil",
    "shear_center_outer_skin_loop",
]
