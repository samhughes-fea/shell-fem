"""Report-only strip K6 vs shell periodic K6 parity (like-conjugate diagonals).

Does **not** gate optimiser constraints. Tolerances follow the CCX acceptance
band (~10% load-aligned). See DEFERRED_RESEARCH.md § Beam→section matching /
F2.1 park decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.section_contract.beam_resultant_vector import (
    SHELL_K6_MODE_ORDER,
    STRIP_OPTIM_ORDER,
)

# Diagonal labels in each conjugate order for the four primary stiffness terms.
_PRIMARY_NAMES = ("N", "My", "Mz", "T")


@dataclass(frozen=True)
class K6ParityReport:
    """Per-station strip vs shell K6 diagonal comparison."""

    station_index: int
    strip_diag_named: dict[str, float]
    shell_diag_named: dict[str, float]
    rel_err_named: dict[str, float]
    within_band: dict[str, bool]
    rel_tol: float

    @property
    def ok(self) -> bool:
        return all(self.within_band.values())


def _diag_named(K6: NDArray[np.float64], order: Sequence[str]) -> dict[str, float]:
    k = np.asarray(K6, dtype=np.float64)
    if k.shape != (6, 6):
        raise ValueError(f"K6 must be (6, 6), got {k.shape}")
    idx = {name: i for i, name in enumerate(order)}
    return {name: float(k[idx[name], idx[name]]) for name in _PRIMARY_NAMES}


def compare_strip_shell_k6(
    strip_k6: NDArray[np.float64],
    shell_k6: NDArray[np.float64],
    *,
    station_index: int = 0,
    rel_tol: float = 0.10,
    strip_order: Sequence[str] = STRIP_OPTIM_ORDER,
    shell_order: Sequence[str] = SHELL_K6_MODE_ORDER,
) -> K6ParityReport:
    """Compare EA / EIy / EIz / GJ-like diagonals by **named** conjugate.

    Strip and shell use different Vy/Vz slot orders; only ``N, My, Mz, T`` are
    compared so the shear-slot swap cannot contaminate the report.
    """
    strip_d = _diag_named(strip_k6, strip_order)
    shell_d = _diag_named(shell_k6, shell_order)
    rel: dict[str, float] = {}
    band: dict[str, bool] = {}
    for name in _PRIMARY_NAMES:
        s = shell_d[name]
        t = strip_d[name]
        denom = max(abs(s), abs(t), 1e-30)
        err = abs(s - t) / denom
        rel[name] = float(err)
        band[name] = bool(err <= float(rel_tol))
    return K6ParityReport(
        station_index=int(station_index),
        strip_diag_named=strip_d,
        shell_diag_named=shell_d,
        rel_err_named=rel,
        within_band=band,
        rel_tol=float(rel_tol),
    )


def format_k6_parity_reports(reports: Sequence[K6ParityReport]) -> str:
    """Human-readable table for logs / diagnostics."""
    lines = [
        f"{'st':>4}  {'name':>4}  {'strip':>12}  {'shell':>12}  {'rel':>8}  ok",
    ]
    for r in reports:
        for name in _PRIMARY_NAMES:
            lines.append(
                f"{r.station_index:4d}  {name:>4}  "
                f"{r.strip_diag_named[name]:12.4e}  "
                f"{r.shell_diag_named[name]:12.4e}  "
                f"{r.rel_err_named[name]:8.3f}  "
                f"{'Y' if r.within_band[name] else 'N'}"
            )
    return "\n".join(lines)


def compare_k6_stacks(
    strip_stack: NDArray[np.float64],
    shell_stack: NDArray[np.float64],
    *,
    rel_tol: float = 0.10,
) -> list[K6ParityReport]:
    """Compare ``(n_s, 6, 6)`` stacks station-wise."""
    s = np.asarray(strip_stack, dtype=np.float64)
    h = np.asarray(shell_stack, dtype=np.float64)
    if s.ndim != 3 or h.ndim != 3 or s.shape != h.shape:
        raise ValueError(f"stack shapes must match (n,6,6); got {s.shape} vs {h.shape}")
    return [
        compare_strip_shell_k6(s[i], h[i], station_index=i, rel_tol=rel_tol)
        for i in range(s.shape[0])
    ]


__all__ = [
    "K6ParityReport",
    "compare_strip_shell_k6",
    "compare_k6_stacks",
    "format_k6_parity_reports",
]
