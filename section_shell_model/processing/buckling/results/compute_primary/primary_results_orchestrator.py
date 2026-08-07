"""Package raw eigen solve states into a primary buckling result set."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.buckling_types import (
    PanelBucklingResult,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.buckling.operations.eigen_states import (
    BucklingEigenSolveState,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.buckling.results.containers import (
    ShellBucklingPrimaryResultSet,
)


class ShellBucklingPrimaryResultsOrchestrator:
    """Primary tier: eigenpairs / modal kinematics (panel classical or section)."""

    def __init__(self, eigen_states: list[BucklingEigenSolveState]) -> None:
        self.eigen_states = list(eigen_states)

    def compute(self) -> ShellBucklingPrimaryResultSet:
        panel_results: list[PanelBucklingResult] = []
        statuses: list[str] = []
        section_lambdas = None
        section_modeshapes = None
        eigen_track = "panel"
        for state in self.eigen_states:
            statuses.append(str(state.status))
            if state.is_section_solve:
                eigen_track = "section"
                section_lambdas = np.asarray(state.lambdas, dtype=float)
                section_modeshapes = np.asarray(state.modeshapes, dtype=float)
            panel_results.append(
                PanelBucklingResult(
                    lambdas=state.lambdas,
                    modeshapes=state.modeshapes,
                    n_modes_found=int(state.n_modes_found),
                    panel_label=str(state.panel_label),
                    panel_index=int(state.panel_index),
                    is_compressive=bool(state.is_compressive),
                )
            )
        overall = "ok"
        if statuses and all(s == "not_computed" for s in statuses):
            overall = "not_computed"
        elif statuses and all(s in ("ok", "skipped") for s in statuses):
            overall = "ok"
        elif any(s not in ("ok", "skipped") for s in statuses):
            overall = "partial"
        return ShellBucklingPrimaryResultSet(
            panel_results=panel_results,
            status=overall,
            diagnostics={
                "n_panels": len(panel_results),
                "panel_status": statuses,
                "eigen_track": eigen_track,
            },
            section_lambdas=section_lambdas,
            section_modeshapes=section_modeshapes,
            eigen_track=eigen_track,
        )
