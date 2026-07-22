"""Bridge mode/traction displacements → secondary-tier resultants for periodic BC.

Production periodic uses K6 modal superposition (not a single constrained
``K q = f`` solve). This module scatters ``d_global`` into elemental ``U_e`` and
delegates resultant recovery to :class:`ShellSecondaryResultsOrchestrator`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_precompute.section_shell_model.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import (
    ShellPanelResultants,
    ShellTheory,
)
from blade_precompute.section_shell_model.processing.results.compute_secondary import (
    ShellSecondaryResultsOrchestrator,
)
from blade_precompute.section_shell_model.processing.results.containers.container_hopper import (
    ShellPrimaryResultSet,
)
from blade_precompute.section_shell_model.processing.results.containers.elemental_results import (
    ShellElementalResults,
)


def scatter_element_displacements(
    cache: ShellFormulationCache,
    d_global: np.ndarray,
) -> list[np.ndarray]:
    """Scatter global DOFs to per-element ``U_e`` (aligned with ``cache.stiffness_objects``)."""
    d = np.asarray(d_global, dtype=float)
    U_e: list[np.ndarray] = []
    for stiff in cache.stiffness_objects:
        g_dofs = stiff.global_dof_indices
        if g_dofs is None:
            raise ValueError(f"element {stiff.element_id}: global_dof_indices not set")
        U_e.append(np.asarray(d[g_dofs], dtype=float))
    return U_e


def nest_resultant_rows_by_panel(
    rows: list[ShellPanelResultants],
    n_panels: int,
) -> list[list[ShellPanelResultants]]:
    """Regroup flat secondary rows into ``list[list[ShellPanelResultants]]`` by panel."""
    nested: list[list[ShellPanelResultants]] = [[] for _ in range(n_panels)]
    for row in rows:
        pi = int(row.panel_index)
        if 0 <= pi < n_panels:
            nested[pi].append(row)
    for panel_rows in nested:
        panel_rows.sort(key=lambda r: int(r.station_index))
    return nested


def recover_resultants_from_displacement(
    d_global: np.ndarray,
    cache: ShellFormulationCache,
    panel_maps: list[Any],
    panels: list[Any],
    *,
    provenance_tag: str = "periodic BC",
    shell_theory: ShellTheory | str | None = None,
    L_x: float = 1.0,
) -> list[list[ShellPanelResultants]]:
    """Secondary-tier recovery from a physical global displacement vector."""
    theory = shell_theory or ShellTheory.FSDT
    theory_s = theory.value if isinstance(theory, ShellTheory) else str(theory)
    U_e = scatter_element_displacements(cache, d_global)
    primary = ShellPrimaryResultSet(elemental_results=ShellElementalResults(U_e=U_e))
    secondary = ShellSecondaryResultsOrchestrator(
        primary=primary,
        cache=cache,
        L_x=float(L_x),
        theory=theory_s,
        panel_meta=None,
        panel_maps=panel_maps,
        panels=panels,
        provenance_tag=provenance_tag,
    ).compute()
    rows = (
        secondary.elemental_results.resultant_rows
        if secondary.elemental_results is not None
        else []
    )
    return nest_resultant_rows_by_panel(rows, len(panels))


def recover_resultants_from_modes(
    u_modes: np.ndarray,
    q_beam: np.ndarray,
    cache: ShellFormulationCache,
    panel_maps: list[Any],
    panels: list[Any],
    *,
    provenance_tag: str = "periodic BC",
    shell_theory: ShellTheory | str | None = None,
    L_x: float = 1.0,
) -> list[list[ShellPanelResultants]]:
    """Recover resultants from ``d = u_modes @ q_beam`` via the secondary orchestrator."""
    d_phys = np.asarray(u_modes, dtype=float) @ np.asarray(q_beam, dtype=float)
    return recover_resultants_from_displacement(
        d_phys,
        cache,
        panel_maps,
        panels,
        provenance_tag=provenance_tag,
        shell_theory=shell_theory,
        L_x=L_x,
    )


__all__ = [
    "nest_resultant_rows_by_panel",
    "recover_resultants_from_displacement",
    "recover_resultants_from_modes",
    "scatter_element_displacements",
]
