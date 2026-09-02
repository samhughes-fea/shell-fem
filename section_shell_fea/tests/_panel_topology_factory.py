"""Test-local panel-topology factory without deprecation noise.

This keeps legacy test ergonomics (`build_shell_panel_topology(...)`) while
the production-facing canonical seam is `shell_panel_topology_from_shell_inputs`.
"""

from __future__ import annotations

import warnings
from typing import Any

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    ShellPanelTopology,
    build_shell_panel_topology as _deprecated_build_shell_panel_topology,
)
from blade_structure_utilities.material_utilities.section_bind import LaminateLike


def build_shell_panel_topology(
    naca_code: str,
    web_x_positions: list[float],
    *,
    chord_m: float = 1.0,
    twist_rad: float = 0.0,
    n_web_samples: int = 20,
    n_cap_samples: int = 80,
    structural_family: str = "C",
    web_alignment: str = "chord_normal",
    skin_thickness: float | None = None,
    web_thickness: float | None = None,
    cap_height: float | None = None,
    skin_lam: LaminateLike | None = None,
    web_lam: LaminateLike | None = None,
    cap_lam: LaminateLike | None = None,
    retain_section: bool = True,
) -> ShellPanelTopology:
    """Compatibility factory for tests that still use the old builder surface."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="build_shell_panel_topology is deprecated;*",
            category=DeprecationWarning,
        )
        return _deprecated_build_shell_panel_topology(
            naca_code=naca_code,
            web_x_positions=web_x_positions,
            chord_m=chord_m,
            twist_rad=twist_rad,
            n_web_samples=n_web_samples,
            n_cap_samples=n_cap_samples,
            structural_family=structural_family,
            web_alignment=web_alignment,
            skin_thickness=skin_thickness,
            web_thickness=web_thickness,
            cap_height=cap_height,
            skin_lam=skin_lam,
            web_lam=web_lam,
            cap_lam=cap_lam,
            retain_section=retain_section,
        )


__all__ = ["build_shell_panel_topology"]

