"""Modular tier-3 recovery operators (in-plane CLT + FSDT interface shear).

In-plane CLT treats the laminate as a **stack of plates**: isotropic panels are
one plate; composites are many. ``InPlaneLaminateRecoveryOperator`` runs the same
per-plate outer/mid/inner recovery for every plate after the coupled ABD solve.

Composes existing ``batched_clt_strains`` / ``ply_strains_stresses_material_frame``
and adds resultant-consistent parabolic transverse-shear recovery at ply
interfaces. See PROCESSING_ARCHITECTURE_PROPOSAL.md open question on interlaminar
recovery depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.material_utilities.laminate import (
    Q_bar_transverse,
    default_transverse_shear_moduli,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellPanelResultants,
)
from .compute_clt_strains import batched_clt_strains
from .compute_ply_stresses import ply_strains_stresses_material_frame
from .provenance import interlaminar_inputs_trusted

SHEAR_RECOVERY_MODE_FSDT_PARABOLIC = "fsdt_interface_parabolic"


@dataclass(frozen=True)
class InPlaneLaminateRecoveryResult:
    """Per-panel batched in-plane CLT recovery (panel row order)."""

    eps0: NDArray[np.float64]
    kappa: NDArray[np.float64]
    trusted_mask: NDArray[np.bool_]
    eps_ply_mat: list[np.ndarray | None]
    sig_ply_mat: list[np.ndarray | None]
    n_ply: int


class InPlaneLaminateRecoveryOperator:
    """Coupled ABD N/M → eps0/kappa → per-plate material-frame plane stresses.

    The laminate is a stack of plates; stresses are evaluated at each plate's
    outer / mid / inner faces (same algorithm for ``n_ply=1`` isotropic and
    multi-ply stacks). Per-row arrays are ``(n_ply, 3, 3)``.
    """

    def recover_panel(
        self,
        rows: list[ShellPanelResultants],
        lam: Any,
    ) -> InPlaneLaminateRecoveryResult:
        eps0, kappa, mask = batched_clt_strains(rows, lam)
        n_rows = len(rows)
        n_ply = len(getattr(lam, "plies", None) or [])
        eps_ply: list[np.ndarray | None] = [None] * n_rows
        sig_ply: list[np.ndarray | None] = [None] * n_rows
        for i in range(n_rows):
            if not mask[i]:
                continue
            eps_mat, sig_mat = ply_strains_stresses_material_frame(
                lam, eps0[i], kappa[i],
            )
            eps_ply[i] = eps_mat
            sig_ply[i] = sig_mat
        return InPlaneLaminateRecoveryResult(
            eps0=eps0,
            kappa=kappa,
            trusted_mask=mask,
            eps_ply_mat=eps_ply,
            sig_ply_mat=sig_ply,
            n_ply=n_ply,
        )


@runtime_checkable
class TransverseShearRecoveryProtocol(Protocol):
    """Recover through-thickness interface transverse shears from Q1/Q2 (+ laminate).

    ``q1``/``q2`` are laminate-frame transverse shear resultants
    (:math:`Q_1,Q_2` along :math:`x_1,x_2`). Secondary rows expose them as
    ``ShellPanelResultants.Q1`` / ``.Q2``.
    """

    recovery_mode: str

    def recover_interfaces(
        self,
        *,
        q1: float,
        q2: float,
        lam: Any,
        thickness_m: float | None = None,
        g13_ply: NDArray[np.float64] | None = None,
        g23_ply: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return ``(n_ply+1, 2)`` = ``[tau_13, tau_23]`` at ply interfaces [Pa].

        ``g13_ply``/``g23_ply`` are an optional caller-supplied
        :func:`_ply_transverse_g` result for this laminate -- every row in a
        panel shares the same laminate, so a caller iterating many rows for one
        panel should compute this once and pass it through rather than have
        each row recompute it from scratch.
        """
        ...


def _ply_transverse_g(lam: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-ply laminate-frame (Q55, Q44) ≈ (G13_eff, G23_eff) for τ13 / τ23."""
    n = len(lam.plies)
    g13 = np.zeros(n, dtype=np.float64)
    g23 = np.zeros(n, dtype=np.float64)
    for k, (ply, ang) in enumerate(lam.plies):
        g23_mat, g13_mat = default_transverse_shear_moduli(
            float(ply.E1),
            float(ply.E2),
            G23=getattr(ply, "G23", None),
            G13=getattr(ply, "G13", None),
        )
        q44, q55 = Q_bar_transverse(float(ang), g23_mat, g13_mat)
        g23[k] = q44
        g13[k] = q55
    return g13, g23


def parabolic_weight(x3: float, h: float) -> float:
    """Free-surface-zero parabola ``1 - (2 x3 / h)^2`` (peak 1 at midplane)."""
    if h <= 0.0:
        return 0.0
    xi = 2.0 * float(x3) / float(h)
    return max(0.0, 1.0 - xi * xi)


def _int_parab(
    x3_a: NDArray[np.float64],
    x3_b: NDArray[np.float64],
    h: float,
) -> NDArray[np.float64]:
    """Analytic per-ply integral of the parabolic weight; vectorises over plies.

    ``∫_{x3_a}^{x3_b} (1-(2 x3/h)^2) dx3 =
    (x3_b-x3_a) - (4/(3 h^2))(x3_b^3 - x3_a^3)`` —
    pure elementwise arithmetic, so this works identically for scalar or array
    ``x3_a``/``x3_b`` (one call over all plies at once, not one call per ply).
    """
    return (x3_b - x3_a) - (4.0 / (3.0 * h * h)) * (x3_b**3 - x3_a**3)


def fsdt_interface_shear_profile(
    *,
    q1: float,
    q2: float,
    x3_iface: NDArray[np.float64],
    h: float,
    g13_ply: NDArray[np.float64] | None = None,
    g23_ply: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Resultant-consistent parabolic (± stiffness-weighted) interface shears.

    ``q1``/``q2`` are laminate-frame transverse shears (:math:`Q_1,Q_2`);
    ``x3_iface`` are interface stations along laminate :math:`x_3`.

    For uniform ``G``, the midplane peak is ``1.5 Q / h`` and free surfaces are
    identically zero. Stiffness weighting uses piecewise-constant ply ``G`` with
    interface values averaged from adjacent plies; the profile is renormalised so
    ``∫ τ dx3 = Q``.

    Returns
    -------
    tau_iface
        ``(n_iface, 2)`` with columns ``[tau_13, tau_23]`` [Pa].
    """
    x3 = np.asarray(x3_iface, dtype=np.float64).ravel()
    n_iface = x3.size
    out = np.zeros((n_iface, 2), dtype=np.float64)
    if h <= 0.0 or n_iface < 2:
        return out

    n_ply = n_iface - 1
    if g13_ply is None or g23_ply is None or len(g13_ply) != n_ply:
        g13 = np.ones(n_ply, dtype=np.float64)
        g23 = np.ones(n_ply, dtype=np.float64)
    else:
        g13 = np.asarray(g13_ply, dtype=np.float64).ravel()
        g23 = np.asarray(g23_ply, dtype=np.float64).ravel()

    # Interface G: free surfaces use adjacent ply; interiors average neighbours.
    # (Vectorised: g13_i[1:-1] is the elementwise average of adjacent plies —
    # same values as the old per-interface Python loop, computed in one call.)
    g13_i = np.empty(n_iface, dtype=np.float64)
    g23_i = np.empty(n_iface, dtype=np.float64)
    g13_i[0], g13_i[-1] = g13[0], g13[-1]
    g23_i[0], g23_i[-1] = g23[0], g23[-1]
    if n_iface > 2:
        g13_i[1:-1] = 0.5 * (g13[:-1] + g13[1:])
        g23_i[1:-1] = 0.5 * (g23[:-1] + g23[1:])

    w = np.clip(1.0 - (2.0 * x3 / h) ** 2, 0.0, None)
    # Integrate piecewise-constant G * parabola over each ply, vectorised over
    # all plies at once (was a per-ply Python accumulation loop).
    dx3_int = _int_parab(x3[:-1], x3[1:], h)
    norm13 = float(np.sum(g13 * dx3_int))
    norm23 = float(np.sum(g23 * dx3_int))

    if norm13 > 0.0:
        out[:, 0] = float(q1) * g13_i * w / norm13
    if norm23 > 0.0:
        out[:, 1] = float(q2) * g23_i * w / norm23
    # Enforce free-surface BC exactly (numerical / weight roundoff).
    out[0, :] = 0.0
    out[-1, :] = 0.0
    return out


class FSDTInterfaceShearRecoveryOperator:
    """FSDT interface τ13/τ23 from Q1,Q2 — parabolic, resultant-consistent, free-surface zero."""

    recovery_mode: str = SHEAR_RECOVERY_MODE_FSDT_PARABOLIC

    def recover_interfaces(
        self,
        *,
        q1: float,
        q2: float,
        lam: Any,
        thickness_m: float | None = None,
        g13_ply: NDArray[np.float64] | None = None,
        g23_ply: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        plies = getattr(lam, "plies", None) or []
        n_ply = len(plies)
        if n_ply == 0:
            return np.zeros((0, 2), dtype=np.float64)
        x3_iface = np.asarray(
            lam.ply_stack_x3_stations().x3_interfaces, dtype=np.float64,
        )
        h = float(thickness_m) if thickness_m is not None and thickness_m > 0.0 else float(
            lam.total_thickness()
        )
        if g13_ply is None or g23_ply is None:
            g13_ply, g23_ply = _ply_transverse_g(lam)
        return fsdt_interface_shear_profile(
            q1=float(q1),
            q2=float(q2),
            x3_iface=x3_iface,
            h=h,
            g13_ply=g13_ply,
            g23_ply=g23_ply,
        )

    def recover_row(
        self,
        row: ShellPanelResultants,
        lam: Any,
    ) -> tuple[NDArray[np.float64] | None, bool]:
        """Recover interfaces for one row; ``(None, True)`` when interlaminar inputs untrusted."""
        if not interlaminar_inputs_trusted(row):
            return None, True
        tau = self.recover_interfaces(
            q1=float(row.Q1),
            q2=float(row.Q2),
            lam=lam,
            thickness_m=float(getattr(row, "thickness_m", 0.0) or 0.0),
        )
        return tau, False


SHEAR_RECOVERY_MODE_CCX_IP = "ccx_ip_fsdt_parabolic"
SHEAR_RECOVERY_MODE_HSDT_Q1 = "hsdt_q1_top_mid_bot"


class CcxIpShearRecoveryOperator(FSDTInterfaceShearRecoveryOperator):
    """CCX path: still recover interfaces from mapped Q1/Q2 (IP stresses → Q upstream).

    CalculiX mid-surface / IP stress mapping already fills ``Q1``/``Q2``
    (:math:`Q_1`/:math:`Q_2`) on ``ShellPanelResultants``; interlaminar FI
    therefore reuses the FSDT parabolic interface profile rather than inventing
    a second through-thickness model.
    """

    recovery_mode: str = SHEAR_RECOVERY_MODE_CCX_IP


class HSDTQ1ShearRecoveryOperator:
    """Parked HSDT ``q1_top/mid/bot`` shear operator (not on optimiser hot path)."""

    recovery_mode: str = SHEAR_RECOVERY_MODE_HSDT_Q1

    def recover_interfaces(
        self,
        *,
        q1: float,
        q2: float,
        lam: Any,
        thickness_m: float | None = None,
        g13_ply: NDArray[np.float64] | None = None,
        g23_ply: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        del q1, q2, thickness_m, g13_ply, g23_ply
        n_ply = len(getattr(lam, "plies", None) or [])
        if n_ply == 0:
            return np.zeros((0, 2), dtype=np.float64)
        # Surfaces + interior interfaces; parked until HSDT element revival.
        return np.zeros((n_ply + 1, 2), dtype=np.float64)

    def recover_row(
        self,
        row: ShellPanelResultants,
        lam: Any,
    ) -> tuple[NDArray[np.float64] | None, bool]:
        if not interlaminar_inputs_trusted(row):
            return None, True
        # Prefer explicit HSDT face/mid summaries when present on the row.
        q1_mid = getattr(row, "q1_mid", None)
        q2_mid = getattr(row, "q2_mid", None)
        if q1_mid is None or q2_mid is None:
            q1_mid = row.Q1
            q2_mid = row.Q2
        if q1_mid is None or q2_mid is None:
            return None, True
        tau = self.recover_interfaces(
            q1=float(q1_mid),
            q2=float(q2_mid),
            lam=lam,
            thickness_m=float(getattr(row, "thickness_m", 0.0) or 0.0),
        )
        return tau, False


def resolve_shear_recovery_operator(
    theory: str | Any | None = None,
) -> TransverseShearRecoveryProtocol:
    """Select the transverse-shear recovery operator for a shell theory label."""
    raw = getattr(theory, "value", None)
    if raw is None:
        raw = getattr(theory, "name", theory)
    key = str(raw or "FSDT").strip().upper()
    if key in {"CCX", "CALCULIX", "CALCULIX_CCX", "S4R"}:
        return CcxIpShearRecoveryOperator()
    if key in {"HSDT"}:
        return HSDTQ1ShearRecoveryOperator()
    return FSDTInterfaceShearRecoveryOperator()


__all__ = [
    "CcxIpShearRecoveryOperator",
    "FSDTInterfaceShearRecoveryOperator",
    "HSDTQ1ShearRecoveryOperator",
    "InPlaneLaminateRecoveryOperator",
    "InPlaneLaminateRecoveryResult",
    "SHEAR_RECOVERY_MODE_CCX_IP",
    "SHEAR_RECOVERY_MODE_FSDT_PARABOLIC",
    "SHEAR_RECOVERY_MODE_HSDT_Q1",
    "TransverseShearRecoveryProtocol",
    "fsdt_interface_shear_profile",
    "parabolic_weight",
    "resolve_shear_recovery_operator",
]
