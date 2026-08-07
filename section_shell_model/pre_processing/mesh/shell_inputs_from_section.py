"""
shell_inputs_from_section
=========================
Adapter that converts a :class:`MultiCellSection` from ``section_geometry``
into a :class:`ShellMeshInputs` payload consumed by the shell-stage
topology generator.

**Pipeline boundary**

This module is the handoff point between the geometry worker
(``section_geometry``) and FE meshing. Prefer constructing the section via
:func:`~blade_structure_utilities.geometry_utilities.section_geometry.build_section_geometry`
/ ``build_section_with_thicknesses`` first. Once :func:`build_shell_mesh_inputs`
returns, no further SDF evaluation (``section[label](x, y)`` / component
``__call__``) should occur on the underlying section — all geometry is
materialised as B-frame polylines inside
:class:`~blade_structure_utilities.geometry_utilities.section_contract.shell_midline_strip.ShellMidlineStrip`
records.

**Naming note:** "caching" a midline here means *materialising the mid-surface
locus as a polyline* for the mesh handoff.  This is distinct from any SDF
*evaluation cache* held internally by the ``section_geometry`` subcomponent
objects.

**Architecture**

Every SDF subcomponent built by ``section_geometry`` (``OuterSkin``,
``ShearWeb``, ``SparCap`` / ``ContinuousSparCap``) exposes a
``midline_polyline()`` method that returns its midline in the **chord frame**.
:func:`build_shell_midline_strips` (in ``section_geometry.interface``) iterates
those, rotates each polyline into the **B-frame**, and returns them as
:class:`~blade_structure_utilities.geometry_utilities.section_contract.shell_midline_strip.ShellMidlineStrip`
records.  This adapter wraps that call and appends LE/TE context to produce
:class:`ShellMeshInputs`.

After skin split, **web** endpoints snap to the skin polyline at their
projected arc-lengths; **spar cap** endpoints then prefer the **nearest**
snapped **shear-web foot** (within a fixed tolerance) so junction clustering
reuses web–skin nodes. Cap **interiors** may then be replaced by intersections
of **inward** rays from master skin vertices with each cap midline (see
:func:`~blade_structure_utilities.geometry_utilities.section_geometry.interface.shell_strip_topology.build_shell_strip_topology`); cap endpoints are not moved by
that step. Finally, :func:`~blade_structure_utilities.geometry_utilities.section_geometry.interface.shell_strip_topology.build_shell_strip_topology` adds vertices at
strict interior cap–web crossings and near web feet (within a proximity
tolerance) so caps carry mesh nodes at skin–web T-junctions along the strip.

**Taxonomy (structural family Y):** **Y = C** (discrete caps per web) yields
cap–web interior junctions at every web by construction. **Y = B** (single cap
band) may have no strict interior cap–web crossing at low web counts, but at
larger **X** (more webs) the band can cross interior webs, so the same insertion
logic applies when geometry demands it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.geometry_utilities.section_contract.shell_midline_strip import ShellMidlineStrip
from blade_structure_utilities.geometry_utilities.section_geometry.interface.shell_midline_export import (
    build_shell_midline_strips,
    rotate_chord_to_blade,
)

from blade_structure_utilities.geometry_utilities.section_geometry.interface.shell_strip_topology import (
    ShellStripTopologySpec,
    build_shell_strip_topology,
    _SKIN_RAY_CAP_BBOX_MARGIN_M,
    _concat_skin_ring_vertices,
    _insert_cap_web_interior_junctions,
    _normal_2d_ccw,
    _pick_inward_ray_direction,
    _point_to_open_polyline_nearest,
    _resample_cap_interiors_from_skin_rays,
    _resample_one_cap_interior_from_skin_rays,
    _skin_ring_vertex_tangents,
    _split_skin_at_junctions,
)

# Near-zero segment-length threshold (metres). Segments shorter than this are
# considered degenerate and will raise ValueError during strip validation.
_MIN_SEGMENT_LENGTH_M: float = 1e-12

# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellMeshInputs:
    """Per-subcomponent midline payload for the shell-stage topology generator.

    All polylines are expressed in the rotated **B-frame** so the downstream
    mesh can be plotted directly on top of ``section_geometry`` outputs
    without further rotation.

    ``leading_edge_b`` / ``trailing_edge_b`` are the airfoil LE/TE points
    rotated into the B-frame and provided as scalar context for diagnostics
    or any consumer that needs to orient the section without recomputing
    the airfoil contour.
    """

    chord_m: float
    twist_rad: float
    layout_key: str | None
    midlines: list[ShellMidlineStrip] = field(default_factory=list)
    leading_edge_b: tuple[float, float] = (0.0, 0.0)
    trailing_edge_b: tuple[float, float] = (1.0, 0.0)


# ---------------------------------------------------------------------------
# Strip validation (private)
# ---------------------------------------------------------------------------


def _validate_strips(strips: tuple[ShellMidlineStrip, ...]) -> None:
    """Raise ``ValueError`` if any strip fails the geometric sanity checks.

    B1: midline_b must have shape ``(N, 2)`` with ``N >= 2``.
    B2: No near-zero consecutive segment lengths (< ``_MIN_SEGMENT_LENGTH_M``).
    """
    for s in strips:
        pts = np.asarray(s.midline_b)

        # B1 — shape and minimum point count.
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                f"Strip {s.label!r} ({s.kind}): midline_b must have shape (N, 2), "
                f"got {pts.shape}."
            )
        if pts.shape[0] < 2:
            raise ValueError(
                f"Strip {s.label!r} ({s.kind}): midline_b must have at least 2 "
                f"vertices, got {pts.shape[0]}."
            )

        # B2 — near-zero segment lengths.
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        bad = np.where(seg_lengths < _MIN_SEGMENT_LENGTH_M)[0]
        if bad.size:
            raise ValueError(
                f"Strip {s.label!r} ({s.kind}): {bad.size} degenerate segment(s) "
                f"with length < {_MIN_SEGMENT_LENGTH_M:.2e} m at vertex indices "
                f"{bad.tolist()}. Check midline_polyline() implementation for "
                f"this subcomponent."
            )

# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------


def _raise_missing_components_error(section: Any, layout_key: str | None) -> None:
    """Raise a descriptive ValueError when ``_components_unrotated`` is absent.

    Distinguishes two cases:
    * ``airfoil_sdf_only`` layout (0A / 0B) — geometry_mode flag detected via
      the layout_key registry or duck-typed from the section class name.
    * Broken multicell object — likely a programming error.
    """
    is_sdf_only = False

    # Try layout registry first (fastest signal when layout_key is provided).
    if layout_key is not None:
        try:
            from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import resolve_system_type

            spec = resolve_system_type(layout_key)
            is_sdf_only = spec.geometry_mode == "airfoil_sdf_only"
        except (KeyError, Exception):
            pass

    # Fallback: duck-type by checking the section class name.
    if not is_sdf_only:
        cls_name = type(section).__name__
        if "OuterSkinOnly" in cls_name or "SdfOnly" in cls_name:
            is_sdf_only = True

    if is_sdf_only:
        raise ValueError(
            f"build_shell_mesh_inputs: section for layout {layout_key!r} uses "
            f"geometry_mode='airfoil_sdf_only' and does not expose "
            f"'_components_unrotated'. Subcomponent midline strips (skin, webs, "
            f"caps) are not available for this layout type. Use a multicell layout "
            f"(e.g. '2D-F', '1A-CN') or call build_shell_midline_strips only on "
            f"MultiCellSection objects."
        )
    raise ValueError(
        f"build_shell_mesh_inputs: section object of type {type(section)!r} does "
        f"not expose '_components_unrotated'. Expected a "
        f"MultiCellSection-compatible object produced by build_section_view with "
        f"a multicell layout. If you intended an airfoil_sdf_only layout (0A/0B), "
        f"pass the correct layout_key so the error message can guide you."
    )


def build_shell_mesh_inputs(
    section: Any,
    *,
    twist_rad: float,
    layout_key: str | None = None,
    n_web_samples: int = 20,
    n_cap_samples: int = 80,
    cap_resample_from_skin_rays: bool = True,
    skin_ray_cap_bbox_margin_m: float = _SKIN_RAY_CAP_BBOX_MARGIN_M,
) -> ShellMeshInputs:
    """Convert a ``MultiCellSection`` into a :class:`ShellMeshInputs` payload.

    Delegates midline extraction and B-frame rotation to
    :func:`~blade_structure_utilities.geometry_utilities.section_geometry.interface.shell_midline_export.build_shell_midline_strips`,
    then appends airfoil LE/TE context.

    Parameters
    ----------
    section
        :class:`blade_structure_utilities.geometry_utilities.section_geometry.engine.sections.multicell.MultiCellSection`
        instance. Must expose:

        * ``airfoil`` — the underlying ``AirfoilSDF`` (used for
          chord length and LE/TE diagnostics);
        * ``_components_unrotated`` — chord-frame component objects
          (skin, webs, ...);
        * ``_spar_cap_components_unrotated`` — chord-frame spar cap
          objects.
    twist_rad
        Section twist in radians.
    layout_key
        Optional ``SystemTypeXY-Z`` key passed through for
        provenance / diagnostics.
    n_web_samples, n_cap_samples
        Number of points used to sample each web / cap midline.
    cap_resample_from_skin_rays
        When ``True`` (default), after :func:`_split_skin_at_junctions`, cap
        strip **interior** vertices are rebuilt from inward skin-normal ray
        hits; cap endpoints stay snapped at skin–web junctions. After that (or
        after split only), :func:`~blade_structure_utilities.geometry_utilities.section_geometry.interface.shell_strip_topology.build_shell_strip_topology` adds any
        missing cap–web T-junction vertices along continuous caps.
    skin_ray_cap_bbox_margin_m
        Axis-aligned bbox inflation (metres) when selecting skin vertices
        opposite each cap for ray casting.
    """
    airfoil_sdf = getattr(section, "airfoil", None)
    if airfoil_sdf is None:
        raise ValueError("section must expose an 'airfoil' attribute (AirfoilSDF).")
    chord_m = float(getattr(airfoil_sdf, "chord", 1.0))

    # B3 — detect airfoil_sdf_only layouts early and raise a targeted error,
    # rather than letting build_shell_midline_strips emit a generic message.
    if getattr(section, "_components_unrotated", None) is None:
        _raise_missing_components_error(section, layout_key)

    strips = build_shell_midline_strips(
        section,
        twist_rad=twist_rad,
        n_web_samples=n_web_samples,
        n_cap_samples=n_cap_samples,
    )
    _validate_strips(strips)
    topology_spec = ShellStripTopologySpec(
        cap_resample_from_skin_rays=cap_resample_from_skin_rays,
        skin_ray_cap_bbox_margin_m=skin_ray_cap_bbox_margin_m,
    )
    strips = build_shell_strip_topology(strips, topology_spec)
    _validate_strips(strips)

    le_S = np.asarray(airfoil_sdf.leading_edge, dtype=float).reshape(1, 2)
    te_S = np.asarray(airfoil_sdf.trailing_edge, dtype=float).reshape(1, 2)
    le_b = tuple(float(v) for v in rotate_chord_to_blade(le_S, twist_rad)[0])
    te_b = tuple(float(v) for v in rotate_chord_to_blade(te_S, twist_rad)[0])

    return ShellMeshInputs(
        chord_m=chord_m,
        twist_rad=float(twist_rad),
        layout_key=layout_key,
        midlines=list(strips),
        leading_edge_b=le_b,  # type: ignore[arg-type]
        trailing_edge_b=te_b,  # type: ignore[arg-type]
    )


__all__ = [
    "ShellMeshInputs",
    "ShellMidlineStrip",
    "ShellStripTopologySpec",
    "build_shell_mesh_inputs",
    "build_shell_strip_topology",
    "_SKIN_RAY_CAP_BBOX_MARGIN_M",
    "_concat_skin_ring_vertices",
    "_insert_cap_web_interior_junctions",
    "_normal_2d_ccw",
    "_pick_inward_ray_direction",
    "_point_to_open_polyline_nearest",
    "_resample_cap_interiors_from_skin_rays",
    "_resample_one_cap_interior_from_skin_rays",
    "_skin_ring_vertex_tangents",
    "_split_skin_at_junctions",
    "_validate_strips",
]
