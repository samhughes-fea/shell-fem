"""Tier 4: Hashin failure indices from tier 3 ply stresses — in-plane + interlaminar modes.

FI-evaluation half of the historical ``api/fi_recovery.hashin_fi_per_ply_from_rows_batch``
(see PROCESSING_ARCHITECTURE_PROPOSAL.md). Pure consumer of ply-resolved stress —
no CLT solve, no laminate side-channel beyond strength lookups.

The transverse-shear/delamination (interlaminar) mode was folded in here from the
standalone ``compute_interlaminar_fi.py`` in Phase 6 (quaternary scope tightening):
same parabolic-peak-shear formula and ``Qx/Qy`` + thickness inputs, unchanged —
relocation, not reformulation. It is theory-independent (works from any shell
path that supplies ``Qx/Qy``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.section_shell_model.pre_processing.containers.types import ShellPanelResultants
from ..compute_tertiary.provenance import interlaminar_inputs_trusted


def interlaminar_allowable_pa(ply: Any) -> float:
    """Interlaminar (delamination) shear allowable [Pa] for a single ply.

    Prefers the real interlaminar shear strength ``S13`` (1-3 / through-thickness
    shear); falls back to in-plane ``S12`` only when ``S13`` is absent (legacy /
    isotropic plies). Returns ``0.0`` when neither is available (caller should
    skip interlaminar FI).
    """
    s13 = float(getattr(ply, "S13", 0.0) or 0.0)
    if s13 > 0.0:
        return s13
    return float(getattr(ply, "S12", 0.0) or 0.0)


def interlaminar_allowable_from_lam(lam: Any) -> float:
    """First-ply interlaminar allowable for a laminate (canonical laminate policy).

    Interlaminar FI uses the **first ply** S13 (S12 fallback). This matches the
    optimiser scalar path and avoids silently taking a more permissive mid-stack
    ply when allowables differ through the thickness.
    """
    if lam is None or not getattr(lam, "plies", None):
        return 0.0
    ply0 = lam.plies[0][0]
    return interlaminar_allowable_pa(ply0)


def ply_strength_vectors_from_lam(
    lam: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-ply ``(Xt, Xc, Yt, Yc, S12)`` strength vectors from a laminate's plies."""
    xt, xc, yt, yc, s12 = [], [], [], [], []
    for oply, _ in lam.plies:
        xt.append(float(getattr(oply, "Xt", 1e9) or 1e9))
        xc.append(float(getattr(oply, "Xc", 1e9) or 1e9))
        yt.append(float(getattr(oply, "Yt", 1e9) or 1e9))
        yc.append(float(getattr(oply, "Yc", 1e9) or 1e9))
        s12.append(float(getattr(oply, "S12", 1e9) or 1e9))
    return (
        np.asarray(xt, dtype=float),
        np.asarray(xc, dtype=float),
        np.asarray(yt, dtype=float),
        np.asarray(yc, dtype=float),
        np.asarray(s12, dtype=float),
    )


def ply_hashin_fi(
    sig_ply: NDArray[np.float64],
    trusted_mask: NDArray[np.bool_],
    *,
    Xt: NDArray[np.float64],
    Xc: NDArray[np.float64],
    Yt: NDArray[np.float64],
    Yc: NDArray[np.float64],
    S12: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Per-row, per-ply Hashin FI from ply-resolved stress ``(n_rows, n_ply, 3)``.

    ``sig_ply`` may be laminate-frame or material-frame stress depending on the
    caller's ply-stress recovery step; this function only evaluates the criterion
    per ply, matching ``materials.failure_criteria.hashin_fi`` element-by-element
    (row, ply) to reproduce the historical batched-loop ordering.
    """
    from blade_structure_utilities.section_materials.failure_criteria import hashin_fi

    n_rows, n_ply_arr, _ = sig_ply.shape
    n_ply = min(n_ply_arr, len(Xt))
    fi_out = np.zeros((n_rows, max(n_ply_arr, 1)), dtype=float)
    needs_review = ~np.asarray(trusted_mask, dtype=bool)
    for ri in np.flatnonzero(trusted_mask):
        for pi in range(n_ply):
            sig3 = np.asarray(sig_ply[ri, pi], dtype=float).ravel()[:3]
            fi_out[ri, pi] = float(
                hashin_fi(
                    sig3.reshape(1, 3),
                    Xt[pi : pi + 1],
                    Xc[pi : pi + 1],
                    Yt[pi : pi + 1],
                    Yc[pi : pi + 1],
                    S12[pi : pi + 1],
                )[0]
            )
    return fi_out, needs_review


def interlaminar_fi_from_row(
    row: ShellPanelResultants,
    *,
    s_il: float,
    peak_factor: float = 1.5,
) -> tuple[float, bool]:
    """Parabolic-peak interlaminar shear FI from ``Qx/Qy`` resultants (single row).

    Moved unchanged from ``compute_interlaminar_fi.py`` (Phase 6 fold) —
    byte-identical formula and inputs.
    """
    if s_il <= 0.0:
        return 0.0, False
    if not interlaminar_inputs_trusted(row):
        return 0.0, True
    t_row = float(getattr(row, "thickness_m", 0.0) or 0.0)
    if t_row <= 0.0:
        return 0.0, True
    qx = float(row.Qx)
    qy = float(row.Qy)
    tau_il = peak_factor * float(np.hypot(qx, qy)) / t_row
    return float(tau_il / s_il), False


def batched_interlaminar_fi(
    qxy: NDArray[np.float64],
    thickness_m: NDArray[np.float64],
    *,
    s_il: float,
    peak_factor: float = 1.5,
    trusted_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Vectorised counterpart of :func:`interlaminar_fi_from_row` for a batch of rows.

    ``qxy`` is ``(n_rows, 2)`` ``[Qx, Qy]``; ``thickness_m`` is ``(n_rows,)``.
    Same parabolic-peak-shear formula, evaluated array-wise instead of per-row —
    used by :class:`ShellQuaternaryResultsOrchestrator` (see
    ``quaternary_results_orchestrator.py``).

    When ``trusted_mask`` is provided, untrusted rows match
    :func:`interlaminar_fi_from_row` (FI = 0, ``needs_review`` = True).
    """
    n_rows = qxy.shape[0]
    if s_il <= 0.0:
        return np.zeros(n_rows, dtype=float), np.zeros(n_rows, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        tau_il = peak_factor * np.hypot(qxy[:, 0], qxy[:, 1]) / thickness_m
        fi_il = tau_il / s_il
    bad = ~np.isfinite(fi_il) | (thickness_m <= 0.0) | ~np.isfinite(qxy[:, 0]) | ~np.isfinite(qxy[:, 1])
    if trusted_mask is not None:
        trusted = np.asarray(trusted_mask, dtype=bool)
        bad = bad | (~trusted)
    fi_il = np.where(bad, 0.0, fi_il)
    return fi_il, bad


def interlaminar_fi_from_interface_tau(
    tau_iface: NDArray[np.float64],
    *,
    s_il: float,
    trusted_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Interlaminar FI from recovered interface ``[tau_13, tau_23]`` arrays.

    ``tau_iface`` is ``(n_rows, n_iface, 2)``. Peak magnitude per row is
    ``max_iface hypot(tau_13, tau_23)``.

    Returns
    -------
    fi_il, needs_review, use_scalar_fallback
        ``use_scalar_fallback`` is True where the interface peak is non-positive
        (e.g. single-ply free surfaces only) so callers can apply the scalar
        ``1.5 |Q| / t`` path for compatibility.
    """
    n_rows = int(tau_iface.shape[0])
    zeros = np.zeros(n_rows, dtype=float)
    false = np.zeros(n_rows, dtype=bool)
    if s_il <= 0.0:
        return zeros, false, false
    with np.errstate(invalid="ignore"):
        if tau_iface.ndim != 3 or tau_iface.shape[1] == 0:
            peak = np.zeros(n_rows, dtype=float)
        else:
            peak = np.max(np.hypot(tau_iface[..., 0], tau_iface[..., 1]), axis=1)
        fi_il = peak / float(s_il)
    trusted = np.ones(n_rows, dtype=bool) if trusted_mask is None else np.asarray(trusted_mask, dtype=bool)
    needs_review = ~trusted
    use_scalar_fallback = trusted & ((~np.isfinite(peak)) | (peak <= 0.0))
    fi_il = np.where(trusted & np.isfinite(fi_il) & (peak > 0.0), fi_il, 0.0)
    return fi_il, needs_review, use_scalar_fallback


def interlaminar_fi_from_row_fsdt(
    row: ShellPanelResultants,
    lam: Any,
    *,
    s_il: float,
    peak_factor: float = 1.5,
    shear_operator: Any | None = None,
) -> tuple[float, bool, str]:
    """Production interlaminar FI: FSDT interface ``τ13/τ23`` with scalar fallback.

    Prefer peak ``hypot(τ13, τ23) / S_il`` from the theory shear operator (default
    FSDT parabolic interfaces). Fall back to scalar ``peak_factor |Q| / t / S_il``
    when the laminate is single-ply / free-surface-only (peak ≤ 0), the operator
    returns no interfaces, or interlaminar inputs are untrusted.

    Returns
    -------
    fi, needs_review, recovery_mode
        ``recovery_mode`` is ``"interface"``, ``"scalar"``, or ``"untrusted"``.
    """
    if s_il <= 0.0:
        return 0.0, False, "scalar"
    if not interlaminar_inputs_trusted(row):
        return 0.0, True, "untrusted"

    from ..compute_tertiary.recovery_operators import FSDTInterfaceShearRecoveryOperator

    op = shear_operator or FSDTInterfaceShearRecoveryOperator()
    n_ply = len(getattr(lam, "plies", None) or [])
    if n_ply >= 1 and hasattr(op, "recover_row"):
        tau, nr = op.recover_row(row, lam)
        if nr:
            return 0.0, True, "untrusted"
        if tau is not None and tau.size > 0:
            fi_arr, nr_arr, need_fb = interlaminar_fi_from_interface_tau(
                np.asarray(tau, dtype=float).reshape(1, -1, 2),
                s_il=s_il,
            )
            if not bool(need_fb[0]):
                return float(fi_arr[0]), bool(nr_arr[0]), "interface"

    fi_scalar, nr_scalar = interlaminar_fi_from_row(
        row, s_il=s_il, peak_factor=peak_factor,
    )
    return fi_scalar, nr_scalar, "scalar"


__all__ = [
    "batched_interlaminar_fi",
    "interlaminar_fi_from_interface_tau",
    "interlaminar_fi_from_row",
    "interlaminar_fi_from_row_fsdt",
    "ply_hashin_fi",
    "ply_strength_vectors_from_lam",
]
