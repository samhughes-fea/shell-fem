"""Shell panel topology geometry: dataclasses and builders.

The theory-agnostic geometry seam shared by shell formulations (FSDT/MITC4,
external CalculiX). :func:`shell_panel_topology_from_shell_inputs` builds the FE
panel topology from already-materialised :class:`ShellMeshInputs`; production
periodic solves in :mod:`.periodic` consume the same ``shell_inputs`` this way.

Split out of the former ``thin_wall_bridge`` god-module; shared shell panel
topology geometry for all shell formulations.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import Any


@_dataclass
class ShellPanelTopology:
    """Output of :func:`build_shell_panel_topology` / ``shell_panel_topology_from_shell_inputs``.

    ``section`` is optional (viz / SDF plots only). The solve path must use
    ``shell_inputs`` polylines — no SDF re-evaluation after the contract handoff.
    """

    shell_inputs: Any
    panels: list
    booms: list
    webs_geom: list
    n_cells: int
    section: Any | None = None


@_dataclass
class TaperedShellPanelTopology:
    """Restricted tapered counterpart to :class:`ShellPanelTopology`."""
    inboard: ShellPanelTopology
    outboard: ShellPanelTopology
    L_x: float


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
    skin_lam: Any | None = None,
    web_lam: Any | None = None,
    cap_lam: Any | None = None,
    retain_section: bool = True,
) -> "ShellPanelTopology":
    """Thin convenience: NACA + webs → geometry worker → mesh inputs → panels.

    Prefer production callers that already have a geometry-worker result and use
    :func:`shell_panel_topology_from_shell_inputs` with
    :func:`build_shell_mesh_inputs` (or midlines from ``build_section_geometry``).

    ``retain_section`` keeps the live SDF section on the topology for viz only;
    the FE solve uses ``shell_inputs`` polylines.
    """
    from blade_structure_utilities.geometry_utilities.section_geometry import AirfoilSDF
    from blade_structure_utilities.geometry_utilities.section_geometry.engine.sections.section import (
        BladeSectionGeometry,
    )
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.shell_inputs_from_section import (
        build_shell_mesh_inputs,
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
    section = BladeSectionGeometry.from_n_web(airfoil_sdf, **section_kwargs)

    shell_inputs = build_shell_mesh_inputs(
        section,
        twist_rad=twist_rad,
        n_web_samples=n_web_samples,
        n_cap_samples=n_cap_samples,
    )
    return shell_panel_topology_from_shell_inputs(
        shell_inputs,
        section=section if retain_section else None,
        skin_lam=skin_lam,
        web_lam=web_lam,
        cap_lam=cap_lam,
    )


def shell_panel_topology_from_shell_inputs(
    shell_inputs: Any,
    *,
    section: Any | None = None,
    skin_lam: Any | None = None,
    web_lam: Any | None = None,
    cap_lam: Any | None = None,
) -> "ShellPanelTopology":
    """Build a :class:`ShellPanelTopology` from already-materialised mesh inputs.

    Contract-first seam: ``shell_inputs`` carries ``ShellMidlineStrip`` polylines.
    ``section`` is optional and only for viz (SDF plots); the FE solve must not
    re-evaluate SDF.
    """
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.topology import build_section_v2

    panels, booms, webs_geom, n_cells = build_section_v2(
        shell_inputs,
        skin_lam=skin_lam,
        web_lam=web_lam,
        cap_lam=cap_lam,
    )
    return ShellPanelTopology(
        shell_inputs=shell_inputs,
        panels=list(panels),
        booms=list(booms),
        webs_geom=list(webs_geom),
        n_cells=int(n_cells),
        section=section,
    )


def build_tapered_shell_panel_topology(
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
    skin_lam: Any | None = None,
    web_lam: Any | None = None,
    cap_lam: Any | None = None,
) -> "TaperedShellPanelTopology":
    """Build matched inboard/outboard shell panel topology sections for a tapered strip."""
    panel_topology_in = build_shell_panel_topology(
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
    panel_topology_out = build_shell_panel_topology(
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

    labels_in = [str(getattr(p, "label", "")) for p in panel_topology_in.panels]
    labels_out = [str(getattr(p, "label", "")) for p in panel_topology_out.panels]
    if labels_in != labels_out:
        raise ValueError("Tapered shell panel topology sections must have identical panel labels and ordering")

    return TaperedShellPanelTopology(inboard=panel_topology_in, outboard=panel_topology_out, L_x=float(L_x))


__all__ = [
    "ShellPanelTopology",
    "TaperedShellPanelTopology",
    "build_shell_panel_topology",
    "shell_panel_topology_from_shell_inputs",
    "build_tapered_shell_panel_topology",
]
