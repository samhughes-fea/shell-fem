"""Demo/test NACA → bound section shell inputs (not a production FE seam)."""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.bound_section_shell_inputs import (
    BoundSectionShellInputs,
    TaperedBoundSectionShellInputs,
    bind_section_shell_inputs,
)
from blade_structure_utilities.material_utilities.section_bind import LaminateLike


def build_demo_bound_section_shell_inputs(
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
) -> BoundSectionShellInputs:
    """NACA + webs → midlines → strips-first mesh inputs → demo-bound zones."""
    from blade_structure_utilities.geometry_utilities.section_geometry import AirfoilSDF
    from blade_structure_utilities.geometry_utilities.section_geometry.sections import (
        MultiCellSection,
    )
    from blade_structure_utilities.geometry_utilities.section_geometry.midlines.shell_midline_export import (
        build_shell_midline_strips,
    )
    from blade_structure_utilities.geometry_utilities.section_geometry.cell_wall_loops import (
        build_cell_wall_loops,
    )
    from blade_structure_utilities.geometry_utilities.section_geometry.junction_preparation.shell_midline_junction_preparation import (
        prepare_shell_midline_junctions,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff.shell_inputs import (
        shell_mesh_inputs_from_strips,
    )

    airfoil_sdf = AirfoilSDF.from_naca(naca_code, n_points=200, chord=chord_m)
    section_kwargs: dict[str, Any] = {
        "web_x_positions": [x * chord_m for x in web_x_positions],
        "web_alignment": web_alignment,
        "structural_family": structural_family,
        "twist_angle": float(twist_rad),
        "core_enabled": True,
    }
    if skin_thickness is not None:
        section_kwargs["skin_thickness"] = float(skin_thickness)
    if web_thickness is not None:
        section_kwargs["web_thickness"] = float(web_thickness)
    if cap_height is not None:
        section_kwargs["cap_height"] = float(cap_height)
    section = MultiCellSection(airfoil_sdf, **section_kwargs)

    raw = build_shell_midline_strips(
        section,
        twist_rad=float(twist_rad),
        n_web_samples=int(n_web_samples),
        n_cap_samples=int(n_cap_samples),
    )
    strips = prepare_shell_midline_junctions(raw)
    from blade_structure_utilities.geometry_utilities.coordinate_frames import (
        naca_to_twisted_pitch_axis_blade_frame,
    )

    le_S = np.asarray(airfoil_sdf.leading_edge, dtype=float).reshape(1, 2)
    te_S = np.asarray(airfoil_sdf.trailing_edge, dtype=float).reshape(1, 2)
    le_b = tuple(
        float(v)
        for v in naca_to_twisted_pitch_axis_blade_frame(
            le_S, float(twist_rad), float(chord_m)
        )[0]
    )
    te_b = tuple(
        float(v)
        for v in naca_to_twisted_pitch_axis_blade_frame(
            te_S, float(twist_rad), float(chord_m)
        )[0]
    )
    shell_inputs = shell_mesh_inputs_from_strips(
        strips,
        chord_m=float(chord_m),
        twist_rad=float(twist_rad),
        leading_edge_twisted_pitch_axis_blade_frame=le_b,  # type: ignore[arg-type]
        trailing_edge_twisted_pitch_axis_blade_frame=te_b,  # type: ignore[arg-type]
        cell_wall_loops=build_cell_wall_loops(strips),
    )
    return bind_section_shell_inputs(
        shell_inputs,
        skin_lam=skin_lam,
        web_lam=web_lam,
        cap_lam=cap_lam,
        allow_demo_bind=True,
    )


def build_demo_multicell_section_for_naca(
    naca_code: str,
    web_x_positions: list[float],
    *,
    chord_m: float = 1.0,
    twist_rad: float = 0.0,
    structural_family: str = "C",
    web_alignment: str = "chord_normal",
    skin_thickness: float | None = None,
    web_thickness: float | None = None,
    cap_height: float | None = None,
):
    """Return a live ``MultiCellSection`` for viz/tests that need SDF polygons."""
    from blade_structure_utilities.geometry_utilities.section_geometry import AirfoilSDF
    from blade_structure_utilities.geometry_utilities.section_geometry.sections import (
        MultiCellSection,
    )

    airfoil_sdf = AirfoilSDF.from_naca(naca_code, n_points=200, chord=chord_m)
    section_kwargs: dict[str, Any] = {
        "web_x_positions": [x * chord_m for x in web_x_positions],
        "web_alignment": web_alignment,
        "structural_family": structural_family,
        "twist_angle": float(twist_rad),
        "core_enabled": True,
    }
    if skin_thickness is not None:
        section_kwargs["skin_thickness"] = float(skin_thickness)
    if web_thickness is not None:
        section_kwargs["web_thickness"] = float(web_thickness)
    if cap_height is not None:
        section_kwargs["cap_height"] = float(cap_height)
    return MultiCellSection(airfoil_sdf, **section_kwargs)


def build_demo_tapered_bound_section_shell_inputs(
    naca_code: str,
    web_x_positions: list[float],
    *,
    chord_inboard_m: float,
    chord_outboard_m: float,
    twist_inboard_rad: float = 0.0,
    twist_outboard_rad: float = 0.0,
    L_x: float = 1.0,
    n_web_samples: int = 20,
    n_cap_samples: int = 80,
    skin_lam: LaminateLike | None = None,
    web_lam: LaminateLike | None = None,
    cap_lam: LaminateLike | None = None,
) -> TaperedBoundSectionShellInputs:
    """Matched inboard/outboard bound inputs for a tapered strip (demo only)."""
    inboard = build_demo_bound_section_shell_inputs(
        naca_code,
        web_x_positions,
        chord_m=chord_inboard_m,
        twist_rad=twist_inboard_rad,
        n_web_samples=n_web_samples,
        n_cap_samples=n_cap_samples,
        skin_lam=skin_lam,
        web_lam=web_lam,
        cap_lam=cap_lam,
    )
    outboard = build_demo_bound_section_shell_inputs(
        naca_code,
        web_x_positions,
        chord_m=chord_outboard_m,
        twist_rad=twist_outboard_rad,
        n_web_samples=n_web_samples,
        n_cap_samples=n_cap_samples,
        skin_lam=skin_lam,
        web_lam=web_lam,
        cap_lam=cap_lam,
    )

    labels_in = [str(getattr(p, "label", "")) for p in inboard.zones]
    labels_out = [str(getattr(p, "label", "")) for p in outboard.zones]
    if labels_in != labels_out:
        raise ValueError(
            "Tapered bound section shell inputs must have identical panel labels and ordering"
        )

    return TaperedBoundSectionShellInputs(inboard=inboard, outboard=outboard, L_x=float(L_x))


__all__ = [
    "build_demo_bound_section_shell_inputs",
    "build_demo_multicell_section_for_naca",
    "build_demo_tapered_bound_section_shell_inputs",
]
