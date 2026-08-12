"""Build ShellPanelBucklingAssessment rows from primary eigenpairs."""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.buckling_types import (
    PanelBucklingResult,
    ShellPanelBucklingAssessment,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.results.containers import (
    ShellBucklingPrimaryResultSet,
    ShellBucklingSecondaryResultSet,
)


def assessment_from_panel_buckling_result(
    raw_result: PanelBucklingResult,
    *,
    station_index: int,
    diagnostics: dict[str, Any] | None = None,
) -> ShellPanelBucklingAssessment:
    """Map a primary eigenpair result to a secondary panel assessment."""
    lambdas = np.asarray(raw_result.lambdas, dtype=float)
    finite = lambdas[np.isfinite(lambdas)]
    load_multiplier_min = float(np.min(finite)) if finite.size else None
    buckling_index = (
        None
        if load_multiplier_min is None or load_multiplier_min <= 0.0
        else float(1.0 / load_multiplier_min)
    )
    governing_mode = int(np.argmin(finite)) if finite.size else None
    status = "ok" if finite.size else "no_modes"
    if diagnostics:
        diag_status = str(diagnostics.get("status", "") or "")
        if diag_status in ("not_computed", "skipped", "failed"):
            status = diag_status
    return ShellPanelBucklingAssessment(
        panel_label=str(raw_result.panel_label),
        panel_index=int(raw_result.panel_index),
        station_index=int(station_index),
        load_multiplier_min=load_multiplier_min,
        load_multipliers=finite,
        n_modes_found=int(raw_result.n_modes_found),
        is_compressive=bool(raw_result.is_compressive),
        buckling_index=buckling_index,
        governing_mode=governing_mode,
        governing_resultant="N11" if bool(raw_result.is_compressive) else None,
        status=status,
        diagnostics=dict(diagnostics or {}),
    )


class ShellBucklingSecondaryResultsOrchestrator:
    """Secondary tier: load multipliers + buckling_index per panel."""

    def __init__(
        self,
        primary: ShellBucklingPrimaryResultSet,
        *,
        station_index: int = 0,
        panel_diagnostics: list[dict[str, Any]] | None = None,
    ) -> None:
        self.primary = primary
        self.station_index = int(station_index)
        self.panel_diagnostics = list(panel_diagnostics or [])

    def compute(self) -> ShellBucklingSecondaryResultSet:
        assessments: list[ShellPanelBucklingAssessment] = []
        for i, raw in enumerate(self.primary.panel_results):
            diag = self.panel_diagnostics[i] if i < len(self.panel_diagnostics) else {}
            assessments.append(
                assessment_from_panel_buckling_result(
                    raw,
                    station_index=self.station_index,
                    diagnostics=diag,
                )
            )
        return ShellBucklingSecondaryResultSet(
            panel_assessments=assessments,
            status=self.primary.status,
            diagnostics={"n_panels": len(assessments)},
        )
