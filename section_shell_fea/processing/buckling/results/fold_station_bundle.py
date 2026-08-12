"""Fold secondary assessments + panel inputs into ShellStationBucklingBundle."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.buckling_types import (
    ShellPanelBucklingAssessment,
    ShellPanelBucklingInput,
    ShellStationBucklingBundle,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellStationProvenance,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.results.containers import (
    ShellBucklingSecondaryResultSet,
)


def fold_station_buckling_bundle(
    *,
    station_id: str,
    z_station: float,
    panel_inputs: Sequence[ShellPanelBucklingInput],
    secondary: ShellBucklingSecondaryResultSet,
    provenance: ShellStationProvenance | None = None,
    station_index: int | None = None,
    metadata: dict | None = None,
) -> ShellStationBucklingBundle:
    """Aggregate secondary assessments into the station handoff DTO."""
    prov = provenance or ShellStationProvenance()
    assessments: tuple[ShellPanelBucklingAssessment, ...] = tuple(secondary.panel_assessments)
    inputs = tuple(panel_inputs)
    if station_index is None and inputs:
        station_index = int(inputs[0].station_index)
    governing = None
    candidates = [a for a in assessments if a.buckling_index is not None]
    if candidates:
        governing = max(candidates, key=lambda a: float(a.buckling_index or -np.inf))
    meta = dict(metadata or {})
    meta.setdefault("n_panels", len(inputs))
    meta.setdefault("secondary_status", secondary.status)
    return ShellStationBucklingBundle(
        station_id=str(station_id),
        station_index=station_index,
        z_station=float(z_station),
        panel_inputs=inputs,
        panel_results=assessments,
        governing_panel_label=None if governing is None else governing.panel_label,
        governing_panel_index=None if governing is None else governing.panel_index,
        governing_buckling_index=None if governing is None else governing.buckling_index,
        analysis_regime=prov.analysis_regime,
        shell_theory=prov.shell_theory,
        provenance=prov,
        metadata=meta,
    )
