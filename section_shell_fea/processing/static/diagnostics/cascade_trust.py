"""Assemble per-tier trust flags into DiagnosticsBin after P→S→T→Q fill."""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.diagnostics_bin import (
    DiagnosticsBin,
)


def _finite_array(a: Any) -> bool:
    if a is None:
        return True
    arr = np.asarray(a)
    if arr.size == 0:
        return True
    return bool(np.all(np.isfinite(arr)))


def fill_cascade_trust(
    diag: DiagnosticsBin,
    *,
    field: Any | None,
    primary_results: Any | None,
    secondary_results: Any | None,
    tertiary_results: Any | None,
    quaternary_results: Any | None,
    solve_residual_rel: float | None = None,
    solve_pivot_ratio: float | None = None,
    solve_quality_ok: bool = True,
    solve_quality_reason: str = "",
) -> DiagnosticsBin:
    """Mutate ``diag`` with solve quality + per-tier ok flags; return it."""
    diag.solve_residual_rel = solve_residual_rel
    diag.solve_pivot_ratio = solve_pivot_ratio
    diag.solve_quality_ok = bool(solve_quality_ok)
    diag.solve_quality_reason = str(solve_quality_reason or "")

    reasons: list[str] = []
    if not diag.solve_quality_ok:
        reasons.append(f"solve:{diag.solve_quality_reason or 'quality_fail'}")

    primary_ok = bool(diag.solve_quality_ok)
    if field is not None:
        if getattr(field, "nan_detected", False):
            primary_ok = False
            reasons.append("primary:nan_detected")
        if getattr(field, "fixed_dof_leak", False):
            primary_ok = False
            reasons.append("primary:fixed_dof_leak")
        d = getattr(field, "d_global", None)
        if d is not None and not _finite_array(d):
            primary_ok = False
            reasons.append("primary:non_finite_d")
    if primary_results is not None and getattr(primary_results, "global_results", None) is not None:
        gr = primary_results.global_results
        if gr.residual is not None and not _finite_array(gr.residual):
            primary_ok = False
            reasons.append("primary:non_finite_residual")
    diag.primary_ok = primary_ok

    secondary_ok = primary_ok
    if secondary_results is not None and getattr(secondary_results, "elemental_results", None):
        rows = list(secondary_results.elemental_results.resultant_rows or [])
        for row in rows:
            if row is None:
                continue
            for name in ("N11", "N22", "N12", "M11", "M22", "M12"):
                val = getattr(row, name, None)
                if val is None:
                    continue
                if not np.isfinite(float(np.asarray(val, dtype=float).ravel()[0])):
                    secondary_ok = False
                    reasons.append(f"secondary:non_finite_{name}")
                    break
            if not secondary_ok:
                break
    diag.secondary_ok = secondary_ok

    tertiary_ok = secondary_ok
    if tertiary_results is not None:
        ply = getattr(tertiary_results, "ply_results", None)
        if ply is not None:
            mask = getattr(ply, "trusted_mask", None)
            sig = getattr(ply, "sig_ply_mat", None)
            if mask is not None and sig is not None:
                m = np.asarray(mask, dtype=bool)
                s = np.asarray(sig, dtype=float)
                if m.any():
                    if not _finite_array(s[m]):
                        tertiary_ok = False
                        reasons.append("tertiary:non_finite_trusted_sigma")
            meta = getattr(tertiary_results, "meta", None) or {}
            if meta.get("abd_lstsq_fallback"):
                tertiary_ok = False
                reasons.append("tertiary:abd_lstsq_fallback")
    diag.tertiary_ok = tertiary_ok

    quaternary_ok = tertiary_ok
    if quaternary_results is not None:
        gov = getattr(quaternary_results, "governing", None) or {}
        fi = gov.get("fi") if isinstance(gov, dict) else None
        # needs_review often flags interlaminar/extreme-fibre gaps while in-plane
        # Hashin still governs; consumers inspect needs_review separately.
        # Poison quaternary only when the governing scalar itself is unusable.
        if fi is not None and not np.isfinite(float(fi)):
            quaternary_ok = False
            reasons.append("quaternary:non_finite_governing_fi")
        if not diag.solve_quality_ok:
            quaternary_ok = False
            if "solve:" not in ";".join(reasons):
                reasons.append("quaternary:solve_quality")
    diag.quaternary_ok = quaternary_ok

    diag.cascade_trust_ok = bool(
        diag.solve_quality_ok
        and diag.primary_ok
        and diag.secondary_ok
        and diag.tertiary_ok
        and diag.quaternary_ok
    )
    if not diag.cascade_trust_ok and not reasons:
        reasons.append("cascade_trust_failed")
    diag.cascade_trust_reason = ";".join(reasons)
    return diag


def require_cascade_trust(results: Any, *, purpose: str = "cascade") -> None:
    """Raise if the cascade bin is not trustworthy for downstream consumers.

    Fatigue / rainflow readers of tertiary stresses and sizing readers of
    quaternary FI should call this (or check ``diagnostics.cascade_trust_ok``)
    before treating results as authoritative.
    """
    diag = getattr(results, "diagnostics", None)
    if diag is None:
        return
    if bool(getattr(diag, "cascade_trust_ok", True)):
        return
    reason = str(getattr(diag, "cascade_trust_reason", "") or "cascade_trust_ok=False")
    raise RuntimeError(f"untrusted shell cascade for {purpose}: {reason}")


__all__ = ["fill_cascade_trust", "require_cascade_trust"]
