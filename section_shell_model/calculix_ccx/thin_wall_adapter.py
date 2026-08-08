"""Quarantined CCX thin-wall (Bredt) global-coupled solve adapter.

Moved from ``pre_processing/element_library/mitc4/utilities/mitc4_solve_orchestrator.py``
(Phase 6 CalculiX quarantine, see docs/PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md) — the
CCX thin-wall bridge does not belong in an MITC4-specific orchestrator module; it
lives here alongside the rest of the CCX backend.

**Scope:** CCX **validation** only (Bredt / thin-wall global-coupled oracle). Not a
production section FE path and not a substitute for the deleted
``api/legacy_clamped.py`` / ``run_section_with_mitc4_shell`` entry. Production FI uses
``run_section_linear_static_from_resultants`` (contour-supported strip); call this adapter directly for
CCX thin-wall checks, not via the production dispatch registry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import ShellPanelResultants
from blade_structure_utilities.mesh_utilities.section_mesh.mesh_topology import NElements
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.thin_wall.section_vlasov import SectionVlasovResult
from .solve import solve_global_coupled_ccx


def ccx_global_coupled_solve_from_thin_wall(
    airfoil: np.ndarray,
    panels: list,
    webs_geom: Any,
    q_tot: list,
    sig_p: list,
    B: float,
    dB_dx: float,
    n_elements_per_panel: NElements | None = None,
    reference_panel_index: int = 0,
    global_bc_mode: str = "contour_support",
    interface_constraint_mode: str = "shared",
    *,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = 2,
    work_dir: str,
    ccx_exe: str = "ccx",
    job_name: str = "section",
) -> tuple[list[list[ShellPanelResultants]], list[dict], ShellPanelResultants | None, SectionVlasovResult]:
    """External ccx solve from thin-wall recovery outputs."""
    from blade_structure_utilities.fem_utilities.section_shell_model.calculix_ccx.thin_wall_loads import (
        prepare_thin_wall_panel_loads,
    )

    vlasov, nx_panels, nxy_panels = prepare_thin_wall_panel_loads(
        airfoil, panels, webs_geom, q_tot, sig_p, B, dB_dx
    )

    all_panel_results, all_panel_diag, _inp = solve_global_coupled_ccx(
        panels,
        nx_panels,
        nxy_panels,
        work_dir=work_dir,
        ccx_exe=ccx_exe,
        job_name=job_name,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        bc_mode=global_bc_mode,
        interface_constraint_mode=interface_constraint_mode,
    )

    ref: ShellPanelResultants | None = None
    if reference_panel_index < len(all_panel_results) and all_panel_results[reference_panel_index]:
        panel_res = all_panel_results[reference_panel_index]
        ref = panel_res[len(panel_res) // 2]
    return all_panel_results, all_panel_diag, ref, vlasov


__all__ = ["ccx_global_coupled_solve_from_thin_wall"]
