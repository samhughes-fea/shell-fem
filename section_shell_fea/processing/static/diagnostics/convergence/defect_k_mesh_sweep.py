"""Defect K mesh sweep: ``shared`` vs ``transformed_basis`` cascade scalars.

Emits a CSV of M+P/S/T/Q metrics across a density ladder for both junction
modes (with production cap ties on). Pair-wise secondary traction residuals
remain informational; primary decision uses cascade drift rates.
"""

from __future__ import annotations

from pathlib import Path

from .cascade_mesh_convergence import (
    DEFAULT_REFINEMENTS,
    CascadeScalars,
    drift,
    series_by_kind,
    sweep_cascade_convergence,
    write_convergence_csv,
)

MODES = ("shared", "transformed_basis")


def run_defect_k_mesh_sweep(
    *,
    refinements: tuple[float, ...] = DEFAULT_REFINEMENTS,
    cap_tie_mode: str = "skin_offset",
    out_csv: str | Path | None = None,
) -> dict[str, list[CascadeScalars]]:
    """Sweep both interface modes; optionally write a combined CSV."""
    by_mode: dict[str, list[CascadeScalars]] = {}
    all_rows: list[CascadeScalars] = []
    for mode in MODES:
        rows = sweep_cascade_convergence(
            refinements,
            cap_tie_mode=cap_tie_mode,
            interface_constraint_mode=mode,
        )
        by_mode[mode] = rows
        all_rows.extend(rows)
    if out_csv is not None:
        write_convergence_csv(out_csv, all_rows)
    return by_mode


def summarize_hashin_drift(
    by_mode: dict[str, list[CascadeScalars]],
) -> dict[str, dict[str, float]]:
    """Per-mode / per-kind Hashin peak drift for promote/keep decisions."""
    out: dict[str, dict[str, float]] = {}
    for mode, rows in by_mode.items():
        series = series_by_kind(rows, "peak_hashin")
        out[mode] = {kind: drift(vals) for kind, vals in series.items()}
    return out


__all__ = [
    "MODES",
    "run_defect_k_mesh_sweep",
    "summarize_hashin_drift",
]
