"""Bound section shell inputs: dataclasses and production bind seam.

The theory-agnostic geometry seam shared by shell formulations (FSDT/MITC4,
external CalculiX). :func:`bind_section_shell_inputs` adapts already-materialised
:class:`ShellStripMeshingInputs` plus frozen material zones into FE-local bound
inputs; periodic solves consume the same ``shell_inputs`` polylines.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from blade_structure_utilities.material_utilities.section_bind import LaminateLike, MaterialZone
from blade_structure_utilities.mesh_utilities.section_mesh.handoff.shell_inputs import (
    ShellStripMeshingInputs,
)


@dataclass(frozen=True)
class BoundSectionShellInputs:
    """Output of :func:`bind_section_shell_inputs`.

    The solve path uses ``shell_inputs`` polylines — no SDF re-evaluation after
    the contract handoff.
    """

    shell_inputs: ShellStripMeshingInputs
    zones: tuple[MaterialZone, ...]
    webs_geom: tuple[Any, ...]
    n_cells: int

    @property
    def panels(self) -> tuple[MaterialZone, ...]:
        """Deprecated alias of :attr:`zones`."""
        warnings.warn(
            "BoundSectionShellInputs.panels is deprecated; use BoundSectionShellInputs.zones",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.zones


@dataclass(frozen=True)
class TaperedBoundSectionShellInputs:
    """Restricted tapered counterpart to :class:`BoundSectionShellInputs`."""

    inboard: BoundSectionShellInputs
    outboard: BoundSectionShellInputs
    L_x: float


def bind_section_shell_inputs(
    shell_inputs: ShellStripMeshingInputs,
    *,
    zones: tuple[MaterialZone, ...] | list[MaterialZone] | None = None,
    webs_geom: tuple[Any, ...] | list[Any] | None = None,
    n_cells: int | None = None,
    skin_lam: LaminateLike | None = None,
    web_lam: LaminateLike | None = None,
    cap_lam: LaminateLike | None = None,
    allow_demo_bind: bool = False,
    section: Any | None = None,
) -> BoundSectionShellInputs:
    """Build :class:`BoundSectionShellInputs` from mesh inputs and frozen zones.

    Contract-first seam: ``shell_inputs`` carries ``ShellMidlineStrip`` polylines.
    The FE solve must not re-evaluate SDF.

    **Production path:** pass frozen ``zones`` (+ optional ``webs_geom`` /
    ``n_cells``) from ``SectionBindRunner`` / orchestration. FE must not own
    assignment policy on the L4 seam.

    **Named demo helper (quarantined):** when ``zones`` is omitted, pass
    ``allow_demo_bind=True`` to call
    ``bind_shell_material_zones(..., allow_demo_laminates=True)`` for tests /
    examples / CCX convenience — not a second L4 sequencer. Omitting zones
    without the flag raises.

    ``section`` is accepted only as a deprecated no-op (visualization context
    must be supplied separately; it is not part of FE solve identity). Removed
    after **2027-03-31**.
    """
    if section is not None:
        warnings.warn(
            "bind_section_shell_inputs(section=...) is ignored; BoundSectionShellInputs "
            "does not carry live geometry. Pass SectionGeometryResult to visualisation "
            "separately (alias expires 2027-03-31).",
            DeprecationWarning,
            stacklevel=2,
        )
    if zones is not None:
        zone_tuple = tuple(zones)
        if webs_geom is None:
            webs_geom = ()
        if n_cells is None:
            n_web = sum(1 for z in zone_tuple if str(getattr(z, "kind", "")) == "web")
            n_cells = int(n_web) + 1
        return BoundSectionShellInputs(
            shell_inputs=shell_inputs,
            zones=zone_tuple,
            webs_geom=tuple(webs_geom),
            n_cells=int(n_cells),
        )

    if not allow_demo_bind:
        raise ValueError(
            "bind_section_shell_inputs requires frozen zones= from "
            "SectionBindRunner / orchestration. Demo-bind is quarantined: pass "
            "allow_demo_bind=True only for tests, examples, or CCX convenience."
        )

    warnings.warn(
        "bind_section_shell_inputs demo-bind path "
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
    return BoundSectionShellInputs(
        shell_inputs=shell_inputs,
        zones=tuple(bound_zones),
        webs_geom=tuple(bound_webs),
        n_cells=int(bound_cells),
    )


__all__ = [
    "BoundSectionShellInputs",
    "TaperedBoundSectionShellInputs",
    "bind_section_shell_inputs",
]
