"""Centralised FEM constants for the MITC4 shell pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MitC4Config:
    # Assembled MITC4+ DOF layout in curvilinear C: u_x, u_s, w, β_s, β_x, β_zeta
    n_dof_node: int = 6
    # Panel MITC4 (no drilling) DOF layout in C: u_x, u_s, w, β_s, β_x
    n_dof_node_panel: int = 5
    # Mindlin/Reissner shear correction factor
    ks: float = 5.0 / 6.0
    # Gauss points for membrane (in-plane) and shear (transverse) integration
    n_gauss_membrane: int = 3
    n_gauss_shear: int = 2
    # Cap panel meshing: polyline deflection (rad) that counts as a kink
    panel_mesh_kink_threshold_rad: float = math.radians(0.35)
    # Cap panel meshing: fraction of nominal linspace step beyond which a
    # cap-geometry sample is added to avoid aliasing smooth junctions
    panel_mesh_s_far_alpha: float = 0.3


MITC4_DEFAULTS = MitC4Config()

__all__ = ["MitC4Config", "MITC4_DEFAULTS"]
