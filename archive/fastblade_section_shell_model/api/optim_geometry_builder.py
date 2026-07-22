"""Build Pipeline-B sections from optimisation geometry (``OptimBladeGeometry``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_precompute.materials.laminate import LaminateDefinition
from blade_precompute.section_definition.types import SectionDefinition
from blade_precompute.section_geometry import AirfoilSDF, MultiCellSection
from blade_precompute.section_optimisation.core.types import OptimBladeGeometry
from blade_precompute.materials.laminate_adapters import LaminateDefinitionMitc4SkinAdapter

from .pipeline_b import (
    PipelineBSection,
    TaperedPipelineBSection,
    pipeline_b_section_from_shell_inputs,
)


def _laminate_for_role(section_def: SectionDefinition, role: str) -> Any | None:
    role_l = role.lower()
    for sub in section_def.subcomponents:
        if role_l in sub.name.lower() and isinstance(sub.material, LaminateDefinition):
            mat = sub.material
            if mat.plies:
                return LaminateDefinitionMitc4SkinAdapter(mat)
    return None


def _first_matching_lam(section_def: SectionDefinition, *keywords: str) -> Any | None:
    for sub in section_def.subcomponents:
        if not isinstance(sub.material, LaminateDefinition):
            continue
        nl = sub.name.lower()
        if any(k in nl for k in keywords):
            mat = sub.material
            if mat.plies:
                return LaminateDefinitionMitc4SkinAdapter(mat)
    return None


def _resolve_laminates(section_def: SectionDefinition) -> tuple[Any | None, Any | None, Any | None]:
    skin = _laminate_for_role(section_def, "skin") or _first_matching_lam(section_def, "skin")
    web = _laminate_for_role(section_def, "web") or _first_matching_lam(section_def, "web")
    cap = (
        _laminate_for_role(section_def, "cap")
        or _first_matching_lam(section_def, "cap", "spar", "boom")
    )
    return skin, web, cap


def _airfoil_sdf_from_station(bg: OptimBladeGeometry, station_index: int) -> AirfoilSDF:
    prof = bg.airfoil_profiles[station_index]
    verts = np.asarray(getattr(prof, "vertices", prof), dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] < 2:
        raise ValueError("airfoil profile must be Nx2")
    chord = float(bg.chord[station_index])
    return AirfoilSDF.from_array(verts[:, :2], chord=chord)


def pipeline_b_section_from_optim_geometry(
    bg: OptimBladeGeometry,
    section_def: SectionDefinition,
    station_index: int,
    *,
    n_web_samples: int = 20,
    n_cap_samples: int = 80,
    structural_family: str | None = None,
) -> PipelineBSection:
    """Build a :class:`PipelineBSection` from one optimisation station.

    Maps per-subcomponent ``LaminateDefinition`` from ``section_def`` onto
    skin / web / cap panel laminates. Uses the station airfoil profile and
    scaled web positions from ``OptimBladeGeometry``.
    """
    from blade_precompute.section_shell_model.pre_processing.mesh.shell_inputs_from_section import (
        build_shell_mesh_inputs,
    )

    i = int(station_index)
    airfoil_sdf = _airfoil_sdf_from_station(bg, i)
    chord = float(bg.chord[i])
    twist_rad = float(np.deg2rad(float(bg.twist[i])))
    wpos = np.asarray(bg.web_positions, dtype=np.float64).ravel()
    # OptimBladeGeometry stores web_positions as (chord_frac - 0.5); MultiCellSection expects [0, chord].
    web_x_abs = [float((xi + 0.5) * chord) for xi in wpos]
    family = structural_family or str(getattr(bg, "structural_family", "C"))
    web_align = (
        "flapwise"
        if str(getattr(bg, "web_orientation", "chord_normal")) == "flapwise"
        else "chord_normal"
    )

    section = MultiCellSection(
        airfoil_sdf,
        web_x_positions=web_x_abs,
        structural_family=family,
        web_alignment=web_align,
        twist_angle=twist_rad,
    )
    shell_inputs = build_shell_mesh_inputs(
        section,
        twist_rad=twist_rad,
        n_web_samples=n_web_samples,
        n_cap_samples=n_cap_samples,
    )
    skin_lam, web_lam, cap_lam = _resolve_laminates(section_def)
    return pipeline_b_section_from_shell_inputs(
        section,
        shell_inputs,
        skin_lam=skin_lam,
        web_lam=web_lam,
        cap_lam=cap_lam,
    )


def build_tapered_pipeline_b_from_optim_stations(
    bg: OptimBladeGeometry,
    section_defs: list[SectionDefinition],
    inboard_index: int,
    outboard_index: int,
    *,
    L_x: float | None = None,
    n_web_samples: int = 20,
    n_cap_samples: int = 80,
    hold_inboard_topology: bool = True,
) -> TaperedPipelineBSection:
    """Matched inboard/outboard pair for tapered periodic interlaminar FI.

    When ``hold_inboard_topology`` is True and outboard topology would differ
    (web count / ply layout), the inboard section topology is reused and only
    chord/twist/profile of the outboard station are interpolated — documented
    approximation for optimisation stations with mismatched topology.
    """
    pb_in = pipeline_b_section_from_optim_geometry(
        bg, section_defs[inboard_index], inboard_index,
        n_web_samples=n_web_samples, n_cap_samples=n_cap_samples,
    )
    try:
        pb_out = pipeline_b_section_from_optim_geometry(
            bg, section_defs[outboard_index], outboard_index,
            n_web_samples=n_web_samples, n_cap_samples=n_cap_samples,
        )
    except Exception:
        if not hold_inboard_topology:
            raise
        pb_out = pb_in

    labels_in = [str(getattr(p, "label", "")) for p in pb_in.panels]
    labels_out = [str(getattr(p, "label", "")) for p in pb_out.panels]
    if labels_in != labels_out:
        if not hold_inboard_topology:
            raise ValueError(
                "Tapered pair requires identical panel labels; "
                f"inboard={labels_in!r} outboard={labels_out!r}"
            )
        pb_out = pb_in

    if L_x is None:
        z = np.asarray(bg.z_stations, dtype=np.float64).ravel()
        dz = abs(float(z[outboard_index]) - float(z[inboard_index]))
        L_x = max(dz, 1e-3)

    return TaperedPipelineBSection(inboard=pb_in, outboard=pb_out, L_x=float(L_x))


__all__ = [
    "build_tapered_pipeline_b_from_optim_stations",
    "pipeline_b_section_from_optim_geometry",
]
