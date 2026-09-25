"""Shell panel topology geometry: dataclasses and builders.

The theory-agnostic geometry seam shared by shell formulations (FSDT/MITC4,
external CalculiX). :func:`shell_panel_topology_from_shell_inputs` builds the FE
panel topology from already-materialised :class:`ShellMeshInputs`; production
periodic solves in :mod:`.periodic` consume the same ``shell_inputs`` this way.

Split out of the former ``thin_wall_bridge`` god-module; shared shell panel
topology geometry for all shell formulations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from blade_structure_utilities.geometry_utilities.section_geometry.midlines.section_midline_host import (
    SectionMidlineHost,
)
from blade_structure_utilities.material_utilities.section_bind import LaminateLike, MaterialZone
from blade_structure_utilities.mesh_utilities.section_mesh.handoff.shell_inputs import (
    ShellMeshInputs,
)


@dataclass(frozen=True)
class ShellPanelTopology:
    """Output of :func:`build_shell_panel_topology` / ``shell_panel_topology_from_shell_inputs``.

``section`` is optional (viz / SDF plots only). The solve path must use
``shell_inputs`` polylines — no SDF re-evaluation after the contract handoff.
    """

    shell_inputs: ShellMeshInputs
    zones: tuple[MaterialZone, ...]
    webs_geom: tuple[Any, ...]
    n_cells: int
    section: SectionMidlineHost | None = None

    @property
    def panels(self) -> tuple[MaterialZone, ...]:
        """Deprecated alias of :attr:`zones`."""
        warnings.warn(
            "ShellPanelTopology.panels is deprecated; use ShellPanelTopology.zones",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.zones


@dataclass(frozen=True)
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
    skin_lam: LaminateLike | None = None,
    web_lam: LaminateLike | None = None,
    cap_lam: LaminateLike | None = None,
    retain_section: bool = True,
) -> ShellPanelTopology:
    """Thin convenience: NACA + webs → midlines → strips-first mesh inputs → panels.

    .. deprecated::
        Prefer production callers that already have a geometry-worker result and
        use :func:`shell_panel_topology_from_shell_inputs` with
        :func:`shell_mesh_inputs_from_geometry_result`.

    ``retain_section`` keeps the live SDF section on the topology for viz only;
    the FE solve uses ``shell_inputs`` polylines.
    """
    warnings.warn(
        "build_shell_panel_topology is deprecated; build ShellMeshInputs via "
        "shell_mesh_inputs_from_geometry_result and call "
        "shell_panel_topology_from_shell_inputs.",
        DeprecationWarning,
        stacklevel=2,
    )
    import numpy as np

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
    from blade_structure_utilities.geometry_utilities.section_geometry.topology.shell_strip_topology import (
        build_shell_strip_topology,
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
    strips = build_shell_strip_topology(raw)
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
    return shell_panel_topology_from_shell_inputs(
        shell_inputs,
        section=section if retain_section else None,
        skin_lam=skin_lam,
        web_lam=web_lam,
        cap_lam=cap_lam,
        allow_demo_bind=True,
    )


def shell_panel_topology_from_shell_inputs(
    shell_inputs: ShellMeshInputs,
    *,
    section: SectionMidlineHost | None = None,
    zones: tuple[MaterialZone, ...] | list[MaterialZone] | None = None,
    webs_geom: tuple[Any, ...] | list[Any] | None = None,
    n_cells: int | None = None,
    skin_lam: LaminateLike | None = None,
    web_lam: LaminateLike | None = None,
    cap_lam: LaminateLike | None = None,
    allow_demo_bind: bool = False,
) -> ShellPanelTopology:
    """Build a :class:`ShellPanelTopology` from already-materialised mesh inputs.

    Contract-first seam: ``shell_inputs`` carries ``ShellMidlineStrip`` polylines.
    ``section`` is optional and only for viz (SDF plots); the FE solve must not
    re-evaluate SDF.

    **Production path:** pass frozen ``zones`` (+ optional ``webs_geom`` /
    ``n_cells``) from ``SectionBindRunner`` / orchestration. FE must not own
    assignment policy on the L4 seam.

    **Named demo helper (quarantined):** when ``zones`` is omitted, pass
    ``allow_demo_bind=True`` to call
    ``bind_shell_material_zones(..., allow_demo_laminates=True)`` for tests /
    examples / CCX convenience — not a second L4 sequencer. Omitting zones
    without the flag raises.
    """
    if zones is not None:
        zone_tuple = tuple(zones)
        if webs_geom is None:
            webs_geom = ()
        if n_cells is None:
            n_web = sum(1 for z in zone_tuple if str(getattr(z, "kind", "")) == "web")
            n_cells = int(n_web) + 1
        return ShellPanelTopology(
            shell_inputs=shell_inputs,
            zones=zone_tuple,
            webs_geom=tuple(webs_geom),
            n_cells=int(n_cells),
            section=section,
        )

    if not allow_demo_bind:
        raise ValueError(
            "shell_panel_topology_from_shell_inputs requires frozen zones= from "
            "SectionBindRunner / orchestration. Demo-bind is quarantined: pass "
            "allow_demo_bind=True only for tests, examples, or CCX convenience."
        )

    warnings.warn(
        "shell_panel_topology_from_shell_inputs demo-bind path "
        "(allow_demo_bind=True) is a named FE leak; prefer frozen zones=.",
        stacklevel=2,
    )

    from blade_structure_utilities.material_utilities.section_bind import bind_shell_material_zones

    bound_zones, bound_webs, bound_cells = bind_shell_material_zones(
        shell_inputs,
        skin_lam=skin_lam,
        web_lam=web_lam,
        cap_lam=cap_lam,
        allow_demo_laminates=True,
    )
    return ShellPanelTopology(
        shell_inputs=shell_inputs,
        zones=tuple(bound_zones),
        webs_geom=tuple(bound_webs),
        n_cells=int(bound_cells),
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
    skin_lam: LaminateLike | None = None,
    web_lam: LaminateLike | None = None,
    cap_lam: LaminateLike | None = None,
) -> TaperedShellPanelTopology:
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

    labels_in = [str(getattr(p, "label", "")) for p in panel_topology_in.zones]
    labels_out = [str(getattr(p, "label", "")) for p in panel_topology_out.zones]
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
