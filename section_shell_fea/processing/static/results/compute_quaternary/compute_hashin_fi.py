"""Tier 4: plane-stress Hashin indices plus FSDT interface-shear diagnostics.

FI-evaluation half of the historical ``api/fi_recovery.hashin_fi_per_ply_from_rows_batch``
(see PROCESSING_ARCHITECTURE_PROPOSAL.md). Pure consumer of ply-resolved stress —
no CLT solve, no laminate side-channel beyond strength lookups.

Interface shear is a diagnostic reconstruction from ``Q1``/``Q2`` (and optional
``tau_iface``). It is not a Hashin mode and is not a delamination criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellPanelResultants
from ..compute_tertiary.provenance import interlaminar_inputs_trusted


@dataclass(frozen=True, slots=True)
class InterfaceShearScreeningResult:
    """Conservative two-sided ply-interface shear screening result.

    This is not a Hashin mode and does not represent ``sigma_33``-driven
    delamination or cohesive damage.
    """

    failure_index: float
    needs_review: bool
    recovery_mode: str
    strength_provenance: str
    governing_interface: int | None = None
    governing_side: str | None = None
    tau13_ratio: float = float("nan")
    tau23_ratio: float = float("nan")


def _material_shear_components(
    tau_element: NDArray[np.float64],
    angle_deg: float,
) -> tuple[float, float]:
    """Rotate ``[tau_13,tau_23]`` from laminate E axes to ply material axes."""
    from blade_structure_utilities.geometry_utilities.coordinate_frames import (
        rotate_transverse_shear_E_to_M,
    )

    tau = np.asarray(tau_element, dtype=np.float64).reshape(2)
    tau13, tau23 = rotate_transverse_shear_E_to_M(tau[0], tau[1], float(angle_deg))
    return float(tau13), float(tau23)


def interface_shear_screening_from_tau(
    tau_iface: NDArray[np.float64],
    lam: Any,
    *,
    trusted: bool = True,
    allow_s12_fallback: bool = False,
    recovery_mode: str = "fsdt_interface_parabolic",
) -> InterfaceShearScreeningResult:
    """Evaluate each interior interface in both adjacent ply material frames.

    The component criterion is
    ``(tau_13/S13)**2 + (tau_23/S23)**2``.  Missing authoritative ``S13`` or
    ``S23`` makes the result unavailable/needs-review; an ``S12`` fallback is
    available only as explicitly labelled compatibility provenance.
    """
    plies = list(getattr(lam, "plies", None) or [])
    tau = np.asarray(tau_iface, dtype=np.float64)
    if not trusted:
        return InterfaceShearScreeningResult(
            float("nan"), True, recovery_mode, "unavailable"
        )
    if tau.shape != (len(plies) + 1, 2) or len(plies) < 2 or not np.all(np.isfinite(tau)):
        return InterfaceShearScreeningResult(
            float("nan"), True, recovery_mode, "unavailable"
        )

    governing: tuple[float, int, str, float, float] | None = None
    provenance = "authoritative_s13_s23"
    for interface in range(1, len(plies)):
        for ply_index, side in ((interface - 1, "lower"), (interface, "upper")):
            ply, angle = plies[ply_index]
            s13 = float(getattr(ply, "S13", 0.0) or 0.0)
            s23 = float(getattr(ply, "S23", 0.0) or 0.0)
            if s13 <= 0.0 or s23 <= 0.0:
                if not allow_s12_fallback:
                    return InterfaceShearScreeningResult(
                        float("nan"),
                        True,
                        recovery_mode,
                        "missing_s13_s23",
                        governing_interface=interface,
                        governing_side=side,
                    )
                fallback = float(getattr(ply, "S12", 0.0) or 0.0)
                if fallback <= 0.0:
                    return InterfaceShearScreeningResult(
                        float("nan"),
                        True,
                        recovery_mode,
                        "missing_strength",
                        governing_interface=interface,
                        governing_side=side,
                    )
                s13 = s13 if s13 > 0.0 else fallback
                s23 = s23 if s23 > 0.0 else fallback
                provenance = "compatibility_s12_fallback"
            tau13, tau23 = _material_shear_components(tau[interface], float(angle))
            ratio13 = tau13 / s13
            ratio23 = tau23 / s23
            failure_index = ratio13 * ratio13 + ratio23 * ratio23
            candidate = (failure_index, interface, side, ratio13, ratio23)
            if governing is None or candidate[0] > governing[0]:
                governing = candidate

    assert governing is not None
    return InterfaceShearScreeningResult(
        failure_index=float(governing[0]),
        needs_review=provenance != "authoritative_s13_s23",
        recovery_mode=recovery_mode,
        strength_provenance=provenance,
        governing_interface=int(governing[1]),
        governing_side=governing[2],
        tau13_ratio=float(governing[3]),
        tau23_ratio=float(governing[4]),
    )


def interlaminar_allowable_pa(ply: Any) -> float:
    """Interface-shear allowable [Pa] for a single ply (diagnostic path).

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
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-ply ``(Xt, Xc, Yt, Yc, S12, S23)`` from laminate plies. No silent defaults."""
    xt, xc, yt, yc, s12, s23 = [], [], [], [], [], []
    for oply, _ in lam.plies:
        xt.append(float(getattr(oply, "Xt", float("nan"))))
        xc.append(float(getattr(oply, "Xc", float("nan"))))
        yt.append(float(getattr(oply, "Yt", float("nan"))))
        yc.append(float(getattr(oply, "Yc", float("nan"))))
        s12.append(float(getattr(oply, "S12", float("nan"))))
        s23.append(float(getattr(oply, "S23", float("nan"))))
    return (
        np.asarray(xt, dtype=float),
        np.asarray(xc, dtype=float),
        np.asarray(yt, dtype=float),
        np.asarray(yc, dtype=float),
        np.asarray(s12, dtype=float),
        np.asarray(s23, dtype=float),
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
    S23: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Per-row, per-ply Hashin envelope FI from ply stress ``(n_rows, n_ply, 3)``.

    ``sig_ply`` **must** be ply material frame **M**
    ``[σ11, σ22, τ12]`` (tertiary ``sig_ply_mat`` mid-face). Feeding **E**-frame
    stress silently under-predicts FI. Matches
    ``materials.failure_criteria.hashin_fi`` element-by-element (row, ply).
    """
    modes, needs_review = ply_hashin_fi_modes(
        sig_ply, trusted_mask, Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12, S23=S23,
    )
    fi_ft, fi_fc, fi_mt, fi_mc = modes
    fi_out = np.maximum(np.maximum(fi_ft, fi_fc), np.maximum(fi_mt, fi_mc))
    return fi_out, needs_review


def ply_hashin_fi_modes(
    sig_ply: NDArray[np.float64],
    trusted_mask: NDArray[np.bool_],
    *,
    Xt: NDArray[np.float64],
    Xc: NDArray[np.float64],
    Yt: NDArray[np.float64],
    Yc: NDArray[np.float64],
    S12: NDArray[np.float64],
    S23: NDArray[np.float64] | None = None,
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.bool_],
]:
    """Per-row, per-ply Hashin mode FIs from ply-resolved stress ``(n_rows, n_ply, 3)``.

    Returns ``((fi_ft, fi_fc, fi_mt, fi_mc), needs_review)`` where each mode array
    is ``(n_rows, n_ply)``.

    Vectorised over both rows and plies in one call: ``hashin_fi_modes`` already
    broadcasts over any leading shape via ``sigma_mat[..., 0]``, so evaluating it
    once on the full ``(n_rows, n_ply, 3)`` batch (with ``Xt``/``Xc``/``Yt``/``Yc``/
    ``S12`` as ``(n_ply,)`` arrays broadcasting against the ``(n_rows, n_ply)``
    output) reproduces the old row-by-row/ply-by-ply Python loop exactly, without
    the O(n_rows * n_ply) call overhead -- the loop form only got away with that
    overhead while every laminate was a single degenerate ply; real multi-ply
    stacks (skin/spar now tens to ~80 plies) make it the dominant per-trial cost.
    """
    from blade_structure_utilities.material_utilities.failure import hashin_fi_modes

    n_rows, n_ply_arr, _ = sig_ply.shape
    n_ply = min(n_ply_arr, len(Xt))
    out_cols = max(n_ply_arr, 1)
    fi_ft = np.zeros((n_rows, out_cols), dtype=float)
    fi_fc = np.zeros((n_rows, out_cols), dtype=float)
    fi_mt = np.zeros((n_rows, out_cols), dtype=float)
    fi_mc = np.zeros((n_rows, out_cols), dtype=float)
    trusted = np.asarray(trusted_mask, dtype=bool)
    needs_review = ~trusted

    if n_rows > 0 and n_ply > 0:
        # Zero out untrusted rows' stress before evaluating so the criterion
        # never sees (possibly garbage/NaN) untrusted stress -- matches the old
        # loop's "skip untrusted rows entirely" behaviour exactly.
        sig_safe = np.where(
            trusted[:, None, None],
            np.asarray(sig_ply[:, :n_ply, :], dtype=float),
            0.0,
        )
        ft, fc, mt, mc = hashin_fi_modes(
            sig_safe,
            Xt[:n_ply],
            Xc[:n_ply],
            Yt[:n_ply],
            Yc[:n_ply],
            S12[:n_ply],
            None if S23 is None else S23[:n_ply],
        )
        row_mask = trusted[:, None]
        fi_ft[:, :n_ply] = np.where(row_mask, ft, 0.0)
        fi_fc[:, :n_ply] = np.where(row_mask, fc, 0.0)
        fi_mt[:, :n_ply] = np.where(row_mask, mt, 0.0)
        fi_mc[:, :n_ply] = np.where(row_mask, mc, 0.0)
    return (fi_ft, fi_fc, fi_mt, fi_mc), needs_review


def interlaminar_fi_from_row(
    row: ShellPanelResultants,
    *,
    s_il: float,
    peak_factor: float = 1.5,
) -> tuple[float, bool]:
    """Parabolic-peak interlaminar shear FI from **E**-frame ``Q1``/``Q2`` (= \\(Q_1,Q_2\\)).

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
    qx = float(row.Q1)
    qy = float(row.Q2)
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

    ``qxy`` is ``(n_rows, 2)`` **E**-frame ``[Q1, Q2]`` (secondary ``Q1``, ``Q2``);
    ``thickness_m`` is ``(n_rows,)``. Same parabolic-peak-shear formula, evaluated
    array-wise — used by :class:`ShellQuaternaryResultsOrchestrator`.

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
    """FSDT interface-shear screen with an explicitly compatible scalar fallback.

    Multi-ply laminates are screened at every interface in both adjacent material
    frames using distinct ``S13``/``S23``.  The scalar ``S_il`` path is retained
    only for legacy single-ply/free-surface compatibility.

    Returns
    -------
    fi, needs_review, recovery_mode
        ``recovery_mode`` is ``"interface"``, ``"compatibility_scalar"``,
        ``"unavailable"``, or ``"untrusted"``.
    """
    if s_il <= 0.0:
        return float("nan"), True, "unavailable"
    if not interlaminar_inputs_trusted(row):
        return 0.0, True, "untrusted"

    from ..compute_tertiary.recovery_operators import FSDTInterfaceShearRecoveryOperator

    op = shear_operator or FSDTInterfaceShearRecoveryOperator()
    n_ply = len(getattr(lam, "plies", None) or [])
    if n_ply >= 1 and hasattr(op, "recover_row"):
        tau, nr = op.recover_row(row, lam)
        if nr:
            return 0.0, True, "untrusted"
        if tau is not None and tau.size > 0 and n_ply >= 2:
            result = interface_shear_screening_from_tau(
                np.asarray(tau, dtype=float),
                lam,
                trusted=True,
                recovery_mode=getattr(op, "recovery_mode", "fsdt_interface_parabolic"),
            )
            if np.isfinite(result.failure_index):
                return result.failure_index, result.needs_review, "interface"
            return float("nan"), True, "unavailable"

    fi_scalar, nr_scalar = interlaminar_fi_from_row(
        row, s_il=s_il, peak_factor=peak_factor,
    )
    return fi_scalar, nr_scalar, "compatibility_scalar"


__all__ = [
    "InterfaceShearScreeningResult",
    "batched_interlaminar_fi",
    "interface_shear_screening_from_tau",
    "interlaminar_fi_from_interface_tau",
    "interlaminar_fi_from_row",
    "interlaminar_fi_from_row_fsdt",
    "ply_hashin_fi",
    "ply_hashin_fi_modes",
    "ply_strength_vectors_from_lam",
]
