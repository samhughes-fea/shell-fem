"""Orchestration-facing periodic solve + Hashin FI workflow."""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_precompute.section_shell_model.pre_processing.containers.types import ShellTheory
from .fi_recovery import (
    hashin_fi_per_ply_from_row,
    max_hashin_fi_from_bundle,
    select_governing_hashin_element,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import ShellSectionStationAnalysis, ShellStationProvenance
from .pipeline_b import PipelineBSection, pipeline_b_section_from_shell_inputs
from .periodic import get_periodic_geometry_state, run_section_periodic_bc


def run_station_periodic_hashin(
    pb: PipelineBSection,
    *,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    traction_field: Any | None = None,
    traction_points_yz: np.ndarray | None = None,
    shell_theory: ShellTheory = ShellTheory.FSDT,
    n_elements_per_panel: int | None = None,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = 2,
) -> dict[str, Any]:
    """Beam loads → periodic unit-cell solve → governing Hashin + provenance audit.

    Returns a dict suitable for orchestration / job JSON:
    ``bundle``, ``max_fi_hashin``, ``needs_review``, ``fi_by_panel``,
    ``provenance_breakdown``.
    """
    bundle = run_section_periodic_bc(
        pb,
        shell_theory=shell_theory,
        N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
        traction_field=traction_field,
        traction_points_yz=traction_points_yz,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
    )
    max_fi, needs_review = max_hashin_fi_from_bundle(bundle)
    fi_by_panel: list[float] = []
    prov_counts: dict[str, int] = {}
    panels = bundle.panels or []
    all_res = bundle.all_panel_results or []
    for pi, panel in enumerate(panels):
        if pi >= len(all_res):
            fi_by_panel.append(0.0)
            continue
        plies = panel.lam.build_plies()
        lam = getattr(panel.lam, "_ld", panel.lam)
        pmax = 0.0
        for row in all_res[pi] or []:
            if row is None:
                continue
            for f, p in row.provenance.items():
                k = p.kind.value
                prov_counts[k] = prov_counts.get(k, 0) + 1
            if hasattr(lam, "plies"):
                fi_v, _ = hashin_fi_per_ply_from_row(row, lam, plies)
                if fi_v.size:
                    pmax = max(pmax, float(np.max(fi_v)))
        fi_by_panel.append(pmax)

    ne = int(n_elements_per_panel) if n_elements_per_panel is not None else 10
    periodic_state = get_periodic_geometry_state(
        pb,
        n_elements_per_panel=ne,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
    )
    K6 = np.asarray(periodic_state[4], dtype=float)

    return {
        "bundle": bundle,
        "K6": K6,
        "max_fi_hashin": max_fi,
        "needs_review": needs_review,
        "fi_by_panel": fi_by_panel,
        "provenance_breakdown": prov_counts,
        "governing_element": select_governing_hashin_element(bundle),
    }


def build_fsdt_periodic_station_analysis(
    section_view: Any,
    shell_inputs: Any,
    *,
    z_station: float,
    N: float = 1.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    n_elements_per_panel: int | None = None,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = 2,
    station_id: str = "station-0",
    provenance: ShellStationProvenance | None = None,
) -> ShellSectionStationAnalysis:
    """Path C periodic solve → canonical ``ShellSectionStationAnalysis`` for recovery prepass."""
    from ..interface.station_analysis_builder import build_station_analysis_from_shell_bundle

    pb = pipeline_b_section_from_shell_inputs(section_view, shell_inputs)
    out = run_station_periodic_hashin(
        pb,
        N=N,
        Vy=Vy,
        Vz=Vz,
        My=My,
        Mz=Mz,
        T=T,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
    )
    bundle = out["bundle"]
    strips = tuple(
        getattr(shell_inputs, "midlines", ())
        or getattr(bundle, "shell_midline_strips", ())
        or ()
    )
    prov = provenance or ShellStationProvenance(
        mesh_n_elements_per_panel=n_elements_per_panel,
        mesh_element_length_frac=(
            None if target_element_length_m is None else float(target_element_length_m)
        ),
    )
    return build_station_analysis_from_shell_bundle(
        z_station=float(z_station),
        strips=strips,
        recovery_bundle=bundle,
        K6=np.asarray(out["K6"], dtype=float),
        station_id=str(station_id),
        provenance=prov,
    )


__all__ = ["build_fsdt_periodic_station_analysis", "run_station_periodic_hashin"]
