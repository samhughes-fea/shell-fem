"""Modular tier-3 recovery operators (in-plane CLT + FSDT interface shear).

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

from blade_precompute.materials.laminate import (
    Q_bar_transverse,
    default_transverse_shear_moduli,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import (
    ShellPanelResultants,
)
from .compute_clt_strains import batched_clt_strains
from .compute_ply_stresses import ply_strains_stresses_material_frame
from .provenance import interlaminar_inputs_trusted

SHEAR_RECOVERY_MODE_FSDT_PARABOLIC = "fsdt_interface_parabolic"


@dataclass(frozen=True)
class InPlaneLaminateRecoveryResult:
    """Per-panel batched in-plane CLT recovery (local panel row order)."""

    eps0: NDArray[np.float64]
    kappa: NDArray[np.float64]
    trusted_mask: NDArray[np.bool_]
    eps_ply_mat: list[np.ndarray | None]
    sig_ply_mat: list[np.ndarray | None]
    n_ply: int


class InPlaneLaminateRecoveryOperator:
    """Coupled ABD N/M → eps0/kappa → material-frame ply membrane/bending stresses."""

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
    """Recover through-thickness interface transverse shears from Qx/Qy (+ laminate)."""

    recovery_mode: str

    def recover_interfaces(
        self,
        *,
        qx: float,
        qy: float,
        lam: Any,
        thickness_m: float | None = None,
    ) -> NDArray[np.float64]:
        """Return ``(n_ply+1, 2)`` = ``[tau_13, tau_23]`` at ply interfaces [Pa]."""
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


def parabolic_weight(z: float, h: float) -> float:
    """Free-surface-zero parabola ``1 - (2z/h)^2`` (peak 1 at midplane)."""
    if h <= 0.0:
        return 0.0
    xi = 2.0 * float(z) / float(h)
    return max(0.0, 1.0 - xi * xi)


def fsdt_interface_shear_profile(
    *,
    qx: float,
    qy: float,
    z_iface: NDArray[np.float64],
    h: float,
    g13_ply: NDArray[np.float64] | None = None,
    g23_ply: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Resultant-consistent parabolic (± stiffness-weighted) interface shears.

    For uniform ``G``, the midplane peak is ``1.5 Q / h`` and free surfaces are
    identically zero. Stiffness weighting uses piecewise-constant ply ``G`` with
    interface values averaged from adjacent plies; the profile is renormalised so
    ``∫ τ dz = Q``.

    Returns
    -------
    tau_iface
        ``(n_iface, 2)`` with columns ``[tau_13, tau_23]`` [Pa].
    """
    z = np.asarray(z_iface, dtype=np.float64).ravel()
    n_iface = z.size
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
    g13_i = np.zeros(n_iface, dtype=np.float64)
    g23_i = np.zeros(n_iface, dtype=np.float64)
    g13_i[0] = g13[0]
    g23_i[0] = g23[0]
    g13_i[-1] = g13[-1]
    g23_i[-1] = g23[-1]
    for i in range(1, n_iface - 1):
        g13_i[i] = 0.5 * (g13[i - 1] + g13[i])
        g23_i[i] = 0.5 * (g23[i - 1] + g23[i])

    w = np.asarray([parabolic_weight(zi, h) for zi in z], dtype=np.float64)
    # Integrate piecewise-constant G * parabola over each ply (Simpson on ends).
    # Analytic ∫_{z_a}^{z_b} (1-(2z/h)^2) dz = (z_b-z_a) - (4/(3 h^2))(z_b^3 - z_a^3)
    def _int_parab(za: float, zb: float) -> float:
        return (zb - za) - (4.0 / (3.0 * h * h)) * (zb**3 - za**3)

    norm13 = 0.0
    norm23 = 0.0
    for k in range(n_ply):
        dz_int = _int_parab(float(z[k]), float(z[k + 1]))
        norm13 += float(g13[k]) * dz_int
        norm23 += float(g23[k]) * dz_int

    if norm13 > 0.0:
        out[:, 0] = float(qx) * g13_i * w / norm13
    if norm23 > 0.0:
        out[:, 1] = float(qy) * g23_i * w / norm23
    # Enforce free-surface BC exactly (numerical / weight roundoff).
    out[0, :] = 0.0
    out[-1, :] = 0.0
    return out


class FSDTInterfaceShearRecoveryOperator:
    """FSDT interface τ13/τ23 from Qx,Qy — parabolic, resultant-consistent, free-surface zero."""

    recovery_mode: str = SHEAR_RECOVERY_MODE_FSDT_PARABOLIC

    def recover_interfaces(
        self,
        *,
        qx: float,
        qy: float,
        lam: Any,
        thickness_m: float | None = None,
    ) -> NDArray[np.float64]:
        plies = getattr(lam, "plies", None) or []
        n_ply = len(plies)
        if n_ply == 0:
            return np.zeros((0, 2), dtype=np.float64)
        z_iface = np.asarray(lam.ply_interfaces_z(), dtype=np.float64)
        h = float(thickness_m) if thickness_m is not None and thickness_m > 0.0 else float(
            lam.total_thickness()
        )
        g13, g23 = _ply_transverse_g(lam)
        return fsdt_interface_shear_profile(
            qx=float(qx),
            qy=float(qy),
            z_iface=z_iface,
            h=h,
            g13_ply=g13,
            g23_ply=g23,
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
            qx=float(row.Qx),
            qy=float(row.Qy),
            lam=lam,
            thickness_m=float(getattr(row, "thickness_m", 0.0) or 0.0),
        )
        return tau, False


SHEAR_RECOVERY_MODE_CCX_IP = "ccx_ip_fsdt_parabolic"
SHEAR_RECOVERY_MODE_HSDT_QX = "hsdt_qx_top_mid_bot"


class CcxIpShearRecoveryOperator(FSDTInterfaceShearRecoveryOperator):
    """CCX path: still recover interfaces from mapped Qx/Qy (IP stresses → Q upstream).

    CalculiX mid-surface / IP stress mapping already fills ``Qx/Qy`` on
    ``ShellPanelResultants``; interlaminar FI therefore reuses the FSDT parabolic
    interface profile rather than inventing a second through-thickness model.
    """

    recovery_mode: str = SHEAR_RECOVERY_MODE_CCX_IP


class HSDTQxShearRecoveryOperator:
    """Parked HSDT ``qx_top/mid/bot`` shear operator (not on optimiser hot path)."""

    recovery_mode: str = SHEAR_RECOVERY_MODE_HSDT_QX

    def recover_interfaces(
        self,
        *,
        qx: float,
        qy: float,
        lam: Any,
        thickness_m: float | None = None,
    ) -> NDArray[np.float64]:
        del qx, qy, thickness_m
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
        qx_mid = getattr(row, "qx_mid", None)
        qy_mid = getattr(row, "qy_mid", None)
        if qx_mid is None or qy_mid is None:
            qx_mid = row.Qx
            qy_mid = row.Qy
        if qx_mid is None or qy_mid is None:
            return None, True
        tau = self.recover_interfaces(
            qx=float(qx_mid),
            qy=float(qy_mid),
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
        return HSDTQxShearRecoveryOperator()
    return FSDTInterfaceShearRecoveryOperator()


__all__ = [
    "CcxIpShearRecoveryOperator",
    "FSDTInterfaceShearRecoveryOperator",
    "HSDTQxShearRecoveryOperator",
    "InPlaneLaminateRecoveryOperator",
    "InPlaneLaminateRecoveryResult",
    "SHEAR_RECOVERY_MODE_CCX_IP",
    "SHEAR_RECOVERY_MODE_FSDT_PARABOLIC",
    "SHEAR_RECOVERY_MODE_HSDT_QX",
    "TransverseShearRecoveryProtocol",
    "fsdt_interface_shear_profile",
    "parabolic_weight",
    "resolve_shear_recovery_operator",
]
