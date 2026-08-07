"""Theory-agnostic assembly from ShellFormulationCache.

Scatter-add (:func:`assemble_section_system`) lives in
:mod:`processing.operations.assemble_global` (R8 / 1d-fem alignment).
This module retains the section-level orchestra:
topology → formulation cache → scatter-add → threaded ABD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.element_library.formulation_factory import (
    build_formulation_cache,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.shell_mesh_map import (
    GlobalNodeMeta,
    NElements,
    PanelGlobalMap,
    collect_global_node_meta,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.operations.assemble_global import (
    assemble_section_system,
    scatter_element_fe,
)

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.connected_panel_topology import (
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


def assemble_section_from_panels(
    panels: list[Any],
    Nx_panels: list[np.ndarray],
    Nxy_panels: list[np.ndarray],
    *,
    panels_outboard: list[Any] | None = None,
    L_x: float = 1.0,
    n_elements_per_panel: NElements | None = None,
    target_element_length_m: float | None = None,
    endpoint_tol: float = 1e-6,
    min_elements_per_panel: int = 2,
    interface_constraint_mode: str = "shared",
    shell_theory: ShellTheory = ShellTheory.FSDT,
    phase_collector: PhaseCollector | None = None,
) -> SectionAssemblyResult:
    """Thin assembly path: topology → formulation cache → scatter-add K/f.

    Used by production periodic K6 caching and by ``assemble_global_shell``.
    Does not apply contour clamps, MPCs, or resultant recovery.
    """
    topology = build_connected_panel_maps(
        panels,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        endpoint_tol=endpoint_tol,
        min_elements_per_panel=min_elements_per_panel,
        interface_constraint_mode=interface_constraint_mode,
    )
    cache = build_formulation_cache(
        topology.panel_maps,
        panels,
        Nx_panels,
        Nxy_panels,
        shell_theory=shell_theory,
        L_x=L_x,
        panels_outboard=panels_outboard,
        phase_collector=phase_collector if phase_collector is not None else _NULL_PHASE_COLLECTOR,
    )
    n_gdof = topology.next_global_node * int(cache.n_dof_node)
    K_global, f_global, _diag = assemble_section_system(cache, n_gdof)
    node_meta = collect_global_node_meta(
        topology.panel_maps, panels, panels_outboard, topology.next_global_node
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


__all__ = [
    "SectionAssemblyResult",
    "assemble_section_from_panels",
    "assemble_section_system",
    "scatter_element_fe",
]
