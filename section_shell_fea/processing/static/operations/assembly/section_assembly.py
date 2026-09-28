"""Theory-agnostic assembly from ShellFormulationCache.

Scatter-add (:func:`assemble_section_system`) lives in
:mod:`processing.operations.assemble_global` (R8 / 1d-fem alignment).
This module retains the section-level orchestra:
topology → formulation cache → scatter-add → threaded ABD.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    AssemblyDiagnostics,
    ShellFormulationCache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.formulation_factory import (
    build_formulation_cache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.shell_mesh_map import (
    GlobalNodeMeta,
    PanelGlobalMap,
    collect_global_node_meta,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.assembly.assemble_global import (
    assemble_section_system,
    scatter_element_fe,
)
from blade_structure_utilities.material_utilities.section_bind import MaterialZone

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.connected_panel_topology import (
    ConnectedPanelTopology,
    build_connected_panel_maps,
)
from .....pre_processing.constitutive.panel_abd import compact_panel_abd_from_stacks
from ...diagnostics.solve.solve_phase_timing import PhaseCollector, _NULL_PHASE_COLLECTOR


@dataclass(frozen=True)
class SectionAssemblyResult:
    """Assembly-only payload: topology, maps, formulation cache, K, and element RHS.

    Does not own supports or MPC construction. ``f_element`` is the scatter of
    element ``Fe`` only; station loads are combined later as
    ``f_total = f_element + f_station``.
    """

    K_global: sp.csr_matrix
    f_element: np.ndarray
    node_meta: GlobalNodeMeta
    panel_maps: list[PanelGlobalMap]
    abd_p: np.ndarray
    t_p: np.ndarray
    g_p: np.ndarray
    lab_p: list[str]
    cache: ShellFormulationCache
    topology: ConnectedPanelTopology
    n_gdof: int
    zones: list[Any]
    diagnostics: AssemblyDiagnostics | None = None
    n_scatter_add: int = 1
    shell_midlines: Any | None = None
    """SG-017 midlines (bonds) for cap-tie membership; optional until compose."""

    @property
    def f_global(self) -> np.ndarray:
        """Deprecated alias for ``f_element`` (element RHS, not station loads)."""
        return self.f_element

    def as_legacy_8tuple(
        self,
    ) -> tuple[sp.csr_matrix, np.ndarray, GlobalNodeMeta, list[PanelGlobalMap], np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Historic assemble_global_shell return shape (without formulation cache)."""
        return (
            self.K_global,
            self.f_element,
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
    shell_midlines: Any | None = None,
) -> SectionAssemblyResult:
    """Thin assembly path: topology → formulation cache → scatter-add K/f.

    Used by the production runner and ``assemble_global_shell``. Does not apply
    contour clamps, MPCs, or resultant recovery. Topology modes affect numbering
    and node metadata only.

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
    mode = str(interface_constraint_mode or "shared")
    if mode == "shared_cluster_frame":
        from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.global_numbering import (
            apply_shared_cluster_frame_to_cache,
        )

        apply_shared_cluster_frame_to_cache(
            cache,
            topology.panel_maps,
            getattr(topology, "endpoint_frame_R", None) or {},
        )
    n_gdof = topology.next_global_node * int(cache.n_dof_node)
    K_global, f_element, diag = assemble_section_system(cache, n_gdof)
    node_meta = collect_global_node_meta(
        topology.panel_maps, zone_list, zones_out_list, topology.next_global_node
    )
    # Virtual cluster-ref geometry only. MPCs/supports are composed later.
    if getattr(topology, "cluster_layer_node", None):
        from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.global_numbering import (
            attach_cluster_node_meta,
        )

        attach_cluster_node_meta(
            node_meta,
            topology.clusters,
            zone_list,
            topology.cluster_layer_node,
            topology.cluster_layer_rot_node,
            None,
        )
    abd_p, t_p, g_p, lab_p = compact_panel_abd_from_stacks(
        topology.panel_maps, cache.panel_abd_stacks
    )
    return SectionAssemblyResult(
        K_global=K_global,
        f_element=f_element,
        node_meta=node_meta,
        panel_maps=topology.panel_maps,
        abd_p=abd_p,
        t_p=t_p,
        g_p=g_p,
        lab_p=lab_p,
        cache=cache,
        topology=topology,
        n_gdof=n_gdof,
        zones=zone_list,
        diagnostics=diag,
        n_scatter_add=1,
        shell_midlines=shell_midlines,
    )


__all__ = [
    "SectionAssemblyResult",
    "assemble_section_from_zones",
    "assemble_section_system",
    "scatter_element_fe",
]
