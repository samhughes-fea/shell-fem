"""Assembly layer for canonical shell station analysis artifacts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from blade_structure_utilities.section_contract.shell_midline_strip import ShellMidlineStrip

from .constitutive_adapter import build_shell_constitutive_input
from .recovery_point_set_builder import build_shell_station_analysis_with_recovery_point_set
from blade_structure_utilities.section_shell_model.pre_processing.containers.types import (
    ShellPanelBucklingAssessment,
    ShellPanelBucklingInput,
    ShellPanelResultants,
    ShellSectionStationAnalysis,
    ShellSectionStationGeometry,
    ShellStationBucklingBundle,
    ShellStationProvenance,
)


def _panel_width_m(panel: Any) -> float:
    s_coords = np.asarray(getattr(panel, "s", ()), dtype=float)
    if s_coords.size >= 2:
        return float(np.max(s_coords) - np.min(s_coords))
    nodes = np.asarray(getattr(panel, "nodes", ()), dtype=float)
    if nodes.ndim == 2 and nodes.shape[0] >= 2:
        seg = np.diff(nodes, axis=0)
        return float(np.sum(np.linalg.norm(seg, axis=1)))
    return 0.0


def _station_index_from_results(panel_results: Sequence[ShellPanelResultants]) -> int:
    if not panel_results:
        return 0
    return int(panel_results[len(panel_results) // 2].station_index)


def _assessment_from_raw_result(
    raw_result: Any,
    *,
    station_index: int,
    diagnostics: dict[str, Any] | None = None,
) -> ShellPanelBucklingAssessment:
    lambdas = np.asarray(getattr(raw_result, "lambdas", ()), dtype=float)
    finite = lambdas[np.isfinite(lambdas)]
    load_multiplier_min = float(np.min(finite)) if finite.size else None
    buckling_index = None if load_multiplier_min is None or load_multiplier_min <= 0.0 else float(1.0 / load_multiplier_min)
    governing_mode = int(np.argmin(finite)) if finite.size else None
    status = "ok" if finite.size else "no_modes"
    return ShellPanelBucklingAssessment(
        panel_label=str(getattr(raw_result, "panel_label", "")),
        panel_index=int(getattr(raw_result, "panel_index", 0)),
        station_index=int(station_index),
        load_multiplier_min=load_multiplier_min,
        load_multipliers=lambdas,
        n_modes_found=int(getattr(raw_result, "n_modes_found", finite.size)),
        is_compressive=bool(getattr(raw_result, "is_compressive", False)),
        buckling_index=buckling_index,
        governing_mode=governing_mode,
        governing_resultant="Nx" if bool(getattr(raw_result, "is_compressive", False)) else None,
        status=status,
        diagnostics=dict(diagnostics or {}),
    )


def _build_station_buckling_bundle(
    *,
    station_id: str,
    z_station: float,
    recovery_bundle: Any,
    provenance: ShellStationProvenance,
) -> ShellStationBucklingBundle | None:
    all_panel_results = list(getattr(recovery_bundle, "all_panel_results", None) or [])
    all_panel_buckling = list(getattr(recovery_bundle, "panel_buckling_results", None) or [])
    panels = list(getattr(recovery_bundle, "panels", None) or [])
    panel_abd = getattr(recovery_bundle, "panel_abd", None)
    panel_thickness = getattr(recovery_bundle, "panel_thickness_m", None)
    panel_labels = list(getattr(recovery_bundle, "panel_labels", None) or [])
    panel_diagnostics = list(getattr(recovery_bundle, "all_panel_diagnostics", None) or [])
    if not all_panel_buckling:
        return None
    if not all_panel_results and not all_panel_buckling:
        return None

    inputs: list[ShellPanelBucklingInput] = []
    results: list[ShellPanelBucklingAssessment] = []
    bundle_station_index: int | None = None
    for pi, panel in enumerate(panels):
        panel_resultants = list(all_panel_results[pi]) if pi < len(all_panel_results) and all_panel_results[pi] else []
        if not panel_resultants:
            continue
        station_index = _station_index_from_results(panel_resultants)
        bundle_station_index = station_index if bundle_station_index is None else bundle_station_index
        label = panel_labels[pi] if pi < len(panel_labels) else str(getattr(panel, "label", "") or f"panel_{pi}")
        source_resultant = panel_resultants[len(panel_resultants) // 2]
        thickness = float(panel_thickness[pi]) if panel_thickness is not None and pi < len(panel_thickness) else float(getattr(panel.lam, "t", 0.0))
        inputs.append(
            ShellPanelBucklingInput(
                panel_label=label,
                panel_index=pi,
                station_index=station_index,
                length_m=1.0,
                width_m=max(_panel_width_m(panel), 1e-12),
                thickness_m=thickness,
                abd=None if panel_abd is None or pi >= len(panel_abd) else np.asarray(panel_abd[pi], dtype=float),
                membrane_resultants=source_resultant.to_N_vec(),
                bending_resultants=source_resultant.to_M_vec(),
                transverse_shear_resultants=np.array([
                    float(source_resultant.Qx or 0.0),
                    float(source_resultant.Qy or 0.0),
                ], dtype=float),
                boundary_condition="simply_supported",
                material_label=str(getattr(getattr(panel, "lam", None), "name", "") or label),
                provenance=provenance,
                source_resultant=source_resultant,
            )
        )
        if pi < len(all_panel_buckling):
            diagnostics = panel_diagnostics[pi] if pi < len(panel_diagnostics) and isinstance(panel_diagnostics[pi], dict) else {}
            results.append(
                _assessment_from_raw_result(
                    all_panel_buckling[pi],
                    station_index=station_index,
                    diagnostics=diagnostics,
                )
            )

    if not inputs and not results:
        return None
    governing = None
    governing_candidates = [res for res in results if res.buckling_index is not None]
    if governing_candidates:
        governing = max(governing_candidates, key=lambda res: float(res.buckling_index or -np.inf))
    return ShellStationBucklingBundle(
        station_id=station_id,
        station_index=bundle_station_index,
        z_station=float(z_station),
        panel_inputs=tuple(inputs),
        panel_results=tuple(results),
        governing_panel_label=None if governing is None else governing.panel_label,
        governing_panel_index=None if governing is None else governing.panel_index,
        governing_buckling_index=None if governing is None else governing.buckling_index,
        analysis_regime=provenance.analysis_regime,
        shell_theory=provenance.shell_theory,
        provenance=provenance,
        metadata={"source_bundle": "mitc4", "n_panels": len(inputs)},
    )


def build_station_analysis_from_shell_bundle(
    *,
    z_station: float,
    strips: Sequence[ShellMidlineStrip],
    recovery_bundle: Any,
    K6: np.ndarray | None = None,
    station_id: str = "station-0",
    strip_material_labels: dict[str, str] | None = None,
    centres: dict[str, float] | None = None,
    inertia_terms: dict[str, float] | None = None,
    warping_metadata: dict[str, Any] | None = None,
    provenance: ShellStationProvenance | None = None,
) -> ShellSectionStationAnalysis:
    """Build canonical shell station analysis from any shell recovery bundle.

    Theory-agnostic: consumes the neutral ``panel_labels`` / ``panel_abd`` /
    ``reference_resultants`` bundle fields, so it works for FSDT (MITC4) and
    CalculiX bundles alike.

    ``K6`` is optional (default ``None``): Saint-Venant periodic homogenisation
    is deprecated; linear-static main-track callers should omit it.
    """

    prov = provenance or ShellStationProvenance()
    geometry = ShellSectionStationGeometry(
        station_id=station_id,
        z_station=float(z_station),
        strips=tuple(strips),
        strip_material_labels=dict(strip_material_labels or {}),
        provenance=prov,
    )
    constitutive = build_shell_constitutive_input(
        z_station=float(z_station),
        K6=None if K6 is None else np.asarray(K6, dtype=float),
        centres=dict(centres or {}),
        inertia_terms=dict(inertia_terms or {}),
        warping_metadata=dict(warping_metadata or {}),
        provenance=prov,
    )
    panel_labels = getattr(recovery_bundle, "panel_labels", None)
    panel_abd = getattr(recovery_bundle, "panel_abd", None)
    ref = getattr(recovery_bundle, "reference_resultants", None)
    all_panel_results = list(getattr(recovery_bundle, "all_panel_results", None) or [])
    shell_resultants_list: list[Any] = []
    if isinstance(ref, (list, tuple)):
        shell_resultants_list.extend(ref)
    elif ref is not None:
        shell_resultants_list.append(ref)
    if not shell_resultants_list:
        for panel_rows in all_panel_results:
            rows = list(panel_rows or [])
            if rows:
                shell_resultants_list.append(rows[len(rows) // 2])
    shell_resultants = tuple(shell_resultants_list)
    station_buckling = getattr(recovery_bundle, "station_buckling", None)
    if station_buckling is None:
        station_buckling = _build_station_buckling_bundle(
            station_id=station_id,
            z_station=float(z_station),
            recovery_bundle=recovery_bundle,
            provenance=prov,
        )
    return build_shell_station_analysis_with_recovery_point_set(
        geometry=geometry,
        constitutive=constitutive,
        panels=recovery_bundle.panels,
        provenance=prov,
        panel_abd=panel_abd,
        panel_labels=panel_labels,
        shell_resultants=shell_resultants,
        buckling=station_buckling,
        recovery_metadata={"source_bundle": "mitc4"},
    )


__all__ = [
    "build_station_analysis_from_shell_bundle",
]
