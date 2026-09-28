"""Test-local bound-input factory without deprecation noise."""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.bound_section_shell_inputs import (
    BoundSectionShellInputs,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.visualisation.demos.demo_bound_section_shell_inputs import (
    build_demo_bound_section_shell_inputs,
    build_demo_multicell_section_for_naca,
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
) -> BoundSectionShellInputs:
    """Compatibility alias for tests that still use the old builder surface."""
    del retain_section  # section no longer carried on bound inputs
    return build_demo_bound_section_shell_inputs(
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
    )


def demo_multicell_section(
    naca_code: str,
    web_x_positions: list[float],
    *,
    chord_m: float = 1.0,
    twist_rad: float = 0.0,
    **kwargs,
):
    """Live section object for tests that need SDF / airfoil polygons."""
    return build_demo_multicell_section_for_naca(
        naca_code,
        web_x_positions,
        chord_m=chord_m,
        twist_rad=twist_rad,
        **kwargs,
    )


def equipollent_section_force_vector(
    panel_topology: BoundSectionShellInputs, resultants, *, L_x: float = 1.0
):
    """Shape Channel-A SFV from bound inputs midlines (test helper)."""
    from blade_structure_utilities.load_utilities.section_loads import (
        equipollent_shape_on_midlines,
    )

    midlines = panel_topology.shell_inputs.midlines
    return equipollent_shape_on_midlines(resultants, midlines, L_x=L_x)


__all__ = [
    "build_shell_panel_topology",
    "demo_multicell_section",
    "equipollent_section_force_vector",
]
