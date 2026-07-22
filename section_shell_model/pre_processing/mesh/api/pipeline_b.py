"""Pipeline B section geometry: dataclasses and builders.

The theory-agnostic geometry seam shared by shell formulations (FSDT/MITC4,
external CalculiX). :func:`pipeline_b_section_from_shell_inputs` builds the FE
panel topology from already-materialised :class:`ShellMeshInputs`; production
periodic solves in :mod:`.periodic` consume the same ``shell_inputs`` this way.

Split out of the former ``thin_wall_bridge`` god-module; shared Pipeline B geometry
for all shell formulations.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import Any


@_dataclass
class PipelineBSection:
    """Output of :func:`build_pipeline_b_section`."""
    section: Any
    shell_inputs: Any
    panels: list
    booms: list
    webs_geom: list
    n_cells: int


@_dataclass
class TaperedPipelineBSection:
    """Restricted tapered counterpart to :class:`PipelineBSection`."""
    inboard: PipelineBSection
    outboard: PipelineBSection
    L_x: float


def build_pipeline_b_section(
    naca_code: str,
    web_x_positions: list[float],
    *,
    chord_m: float = 1.0,
    twist_rad: float = 0.0,
    n_web_samples: int = 20,
    n_cap_samples: int = 80,
    skin_lam: Any | None = None,
    web_lam: Any | None = None,
    cap_lam: Any | None = None,
) -> "PipelineBSection":
    """Build Pipeline B panels from a NACA 4-digit code and web positions.

    Uses ``section_geometry`` (SDF-based) to construct a :class:`MultiCellSection`,
    then calls ``build_shell_mesh_inputs`` and ``build_section_v2`` to produce
    junction-consistent panels matching the ``repro_section_strip_mesh`` topology.

    Parameters
    ----------
    naca_code
        Four-digit NACA code string, e.g. ``"2412"``.
    web_x_positions
        Chord-normalised x-positions of shear webs (empty list for zero webs).
    chord_m
        Chord length [m] for the SDF.
    twist_rad
        Twist angle [rad] for the B-frame rotation.
    """
    from blade_structure_utilities.section_geometry import AirfoilSDF, MultiCellSection
    from blade_structure_utilities.section_shell_model.pre_processing.mesh.shell_inputs_from_section import (
        build_shell_mesh_inputs,
    )
    from blade_structure_utilities.section_shell_model.pre_processing.mesh.topology import build_section_v2

    airfoil_sdf = AirfoilSDF.from_naca(naca_code, n_points=200, chord=chord_m)
    # web_x_positions are chord-normalised fractions; scale to absolute coords for MultiCellSection.
    web_x_abs = [x * chord_m for x in web_x_positions]
    section = MultiCellSection(
        airfoil_sdf,
        web_x_positions=web_x_abs,
        structural_family="C",
    )

    shell_inputs = build_shell_mesh_inputs(
        section,
        twist_rad=twist_rad,
        n_web_samples=n_web_samples,
        n_cap_samples=n_cap_samples,
    )
    return pipeline_b_section_from_shell_inputs(
        section, shell_inputs, skin_lam=skin_lam, web_lam=web_lam, cap_lam=cap_lam,
    )


def pipeline_b_section_from_shell_inputs(
    section: Any,
    shell_inputs: Any,
    *,
    skin_lam: Any | None = None,
    web_lam: Any | None = None,
    cap_lam: Any | None = None,
) -> "PipelineBSection":
    """Build a :class:`PipelineBSection` from already-materialised mesh inputs.

    This is the **theory-agnostic** seam: the caller supplies the section and the
    :class:`ShellMeshInputs` (midline strips) built once via ``build_section_view``
    + ``build_shell_mesh_inputs``, and only the FE panel topology
    (:func:`build_section_v2`) is derived here.  The FSDT (and opt-in CalculiX)
    solves consume the **same** ``shell_inputs`` this way — no per-theory geometry rebuild.
    """
    from blade_structure_utilities.section_shell_model.pre_processing.mesh.topology import build_section_v2

    panels, booms, webs_geom, n_cells = build_section_v2(
        shell_inputs,
        skin_lam=skin_lam,
        web_lam=web_lam,
        cap_lam=cap_lam,
    )
    return PipelineBSection(
        section=section,
        shell_inputs=shell_inputs,
        panels=list(panels),
        booms=list(booms),
        webs_geom=list(webs_geom),
        n_cells=int(n_cells),
    )


def build_tapered_pipeline_b_section(
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
) -> "TaperedPipelineBSection":
    """Build matched inboard/outboard Pipeline-B sections for a tapered strip."""
    pb_in = build_pipeline_b_section(
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
    pb_out = build_pipeline_b_section(
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

    labels_in = [str(getattr(p, "label", "")) for p in pb_in.panels]
    labels_out = [str(getattr(p, "label", "")) for p in pb_out.panels]
    if labels_in != labels_out:
        raise ValueError("Tapered Pipeline-B sections must have identical panel labels and ordering")

    return TaperedPipelineBSection(inboard=pb_in, outboard=pb_out, L_x=float(L_x))


__all__ = [
    "PipelineBSection",
    "TaperedPipelineBSection",
    "build_pipeline_b_section",
    "pipeline_b_section_from_shell_inputs",
    "build_tapered_pipeline_b_section",
]
