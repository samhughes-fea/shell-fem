"""Theory-agnostic assembly from ShellFormulationCache.

Scatter-add (:func:`assemble_section_system`) lives in
:mod:`processing.operations.assemble_global` (R8 / 1d-fem alignment).
This module retains the section-level orchestra:
topology → formulation cache → scatter-add → threaded ABD.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.formulation_factory import (
    build_formulation_cache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import (
    GlobalNodeMeta,
    PanelGlobalMap,
    collect_global_node_meta,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.assemble_global import (
    assemble_section_system,
    scatter_element_fe,
)
from blade_structure_utilities.material_utilities.section_bind import MaterialZone

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.connected_panel_topology import (
    ConnectedPanelTopology,
    build_connected_panel_maps,
)
from ....pre_processing.constitutive.panel_abd import compact_panel_abd_from_stacks
from ..diagnostics.solve_phase_timing import PhaseCollector, _NULL_PHASE_COLLECTOR


@dataclass(frozen=True)
class SectionAssemblyResult:
    """Assembly-only payload: global system + geometry maps + threaded ABD + cache."""

    K_global: sp.csr_matrix
    f_global: np.ndarray
    node_meta: GlobalNodeMeta
    panel_maps: list[PanelGlobalMap]
    abd_p: np.ndarray
    t_p: np.ndarray
    g_p: np.ndarray
    lab_p: list[str]
    cache: ShellFormulationCache
    topology: ConnectedPanelTopology
    n_gdof: int

    def as_legacy_8tuple(
        self,
    ) -> tuple[sp.csr_matrix, np.ndarray, GlobalNodeMeta, list[PanelGlobalMap], np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Historic assemble_global_shell return shape (without formulation cache)."""
        return (
            self.K_global,
            self.f_global,
            self.node_meta,
            self.panel_maps,
            self.abd_p,
            self.t_p,
            self.g_p,
            self.lab_p,
        )


def assemble_section_from_zones(
    zones: Sequence[MaterialZone] | list[Any],
    Nx_panels: list[np.ndarray],
    Nxy_panels: list[np.ndarray],
    *,
    section_strip_mesh: Any,
    zones_outboard: Sequence[MaterialZone] | list[Any] | None = None,
    L_x: float = 1.0,
    endpoint_tol: float = 1e-6,
    interface_constraint_mode: str = "shared",
    shell_theory: ShellTheory = ShellTheory.FSDT,
    phase_collector: PhaseCollector | None = None,
) -> SectionAssemblyResult:
    """Thin assembly path: topology → formulation cache → scatter-add K/f.

    Used by production periodic K6 caching and by ``assemble_global_shell``.
    Does not apply contour clamps, MPCs, or resultant recovery.

    ``section_strip_mesh`` is required (BL-005); density is owned by
    ``mesh_utilities.section_mesh``.
    """
    zone_list = list(zones)
    zones_out_list = list(zones_outboard) if zones_outboard is not None else None
    topology = build_connected_panel_maps(
        zone_list,
        endpoint_tol=endpoint_tol,
        interface_constraint_mode=interface_constraint_mode,
        section_strip_mesh=section_strip_mesh,
    )
    cache = build_formulation_cache(
        topology.panel_maps,
        zone_list,
        Nx_panels,
        Nxy_panels,
        shell_theory=shell_theory,
        L_x=L_x,
        zones_outboard=zones_out_list,
        phase_collector=phase_collector if phase_collector is not None else _NULL_PHASE_COLLECTOR,
    )
    n_gdof = topology.next_global_node * int(cache.n_dof_node)
    K_global, f_global, _diag = assemble_section_system(cache, n_gdof)
    node_meta = collect_global_node_meta(
        topology.panel_maps, zone_list, zones_out_list, topology.next_global_node
    )
    abd_p, t_p, g_p, lab_p = compact_panel_abd_from_stacks(
        topology.panel_maps, cache.panel_abd_stacks
    )
    return SectionAssemblyResult(
        K_global=K_global,
        f_global=f_global,
        node_meta=node_meta,
        panel_maps=topology.panel_maps,
        abd_p=abd_p,
        t_p=t_p,
        g_p=g_p,
        lab_p=lab_p,
        cache=cache,
        topology=topology,
        n_gdof=n_gdof,
    )


def assemble_section_from_panels(
    *args: Any,
    panels: Sequence[MaterialZone] | list[Any] | None = None,
    panels_outboard: Sequence[MaterialZone] | list[Any] | None = None,
    **kwargs: Any,
) -> SectionAssemblyResult:
    """Deprecated alias of :func:`assemble_section_from_zones`.

    Accepts legacy positional ``panels`` or keyword ``panels=`` /
    ``panels_outboard=``.
    """
    warnings.warn(
        "assemble_section_from_panels is deprecated; use assemble_section_from_zones "
        "(and zones= / zones_outboard= kwargs).",
        DeprecationWarning,
        stacklevel=2,
    )
    zones = panels
    if zones is None and args:
        zones = args[0]
        args = args[1:]
    if zones is None:
        raise TypeError("assemble_section_from_panels requires zones/panels")
    if "zones_outboard" not in kwargs and panels_outboard is not None:
        kwargs["zones_outboard"] = panels_outboard
    if "zones_outboard" not in kwargs and "panels_outboard" in kwargs:
        kwargs["zones_outboard"] = kwargs.pop("panels_outboard")
    return assemble_section_from_zones(zones, *args, **kwargs)


__all__ = [
    "SectionAssemblyResult",
    "assemble_section_from_zones",
    "assemble_section_from_panels",
    "assemble_section_system",
    "scatter_element_fe",
]
