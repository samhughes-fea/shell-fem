"""Independent simply-supported isotropic FSDT Navier reference solution."""

from __future__ import annotations

import numpy as np


def simply_supported_uniform_pressure_fsdt(
    *,
    length_x: float,
    length_y: float,
    bending_stiffness: float,
    poisson: float,
    shear_stiffness: float,
    pressure: float,
    modes: int = 3,
    x: float | None = None,
    y: float | None = None,
) -> float:
    """Return Navier deflection from an odd ``modes × modes`` FSDT expansion.

    Each mode solves an independent 3×3 system for
    ``[W_mn, theta1_mn, theta2_mn]``.  This implementation intentionally does
    not import element operators and is therefore a verification oracle.
    """
    lx = float(length_x)
    ly = float(length_y)
    d = float(bending_stiffness)
    a_s = float(shear_stiffness)
    if min(lx, ly, d, a_s) <= 0.0 or int(modes) < 1:
        raise ValueError("lengths, stiffnesses and modes must be positive")
    x_eval = 0.5 * lx if x is None else float(x)
    y_eval = 0.5 * ly if y is None else float(y)
    d12 = float(poisson) * d
    d66 = 0.5 * (1.0 - float(poisson)) * d
    deflection = 0.0
    odd_modes = range(1, 2 * int(modes), 2)
    for m in odd_modes:
        alpha = m * np.pi / lx
        for n in odd_modes:
            beta = n * np.pi / ly
            modal_matrix = np.array(
                [
                    [
                        a_s * (alpha * alpha + beta * beta),
                        -a_s * alpha,
                        -a_s * beta,
                    ],
                    [
                        -a_s * alpha,
                        d * alpha * alpha + d66 * beta * beta + a_s,
                        (d12 + d66) * alpha * beta,
                    ],
                    [
                        -a_s * beta,
                        (d12 + d66) * alpha * beta,
                        d * beta * beta + d66 * alpha * alpha + a_s,
                    ],
                ],
                dtype=float,
            )
            q_mn = 16.0 * float(pressure) / (m * n * np.pi**2)
            modal = np.linalg.solve(modal_matrix, np.array([q_mn, 0.0, 0.0]))
            deflection += (
                modal[0]
                * np.sin(alpha * x_eval)
                * np.sin(beta * y_eval)
            )
    return float(deflection)


def simply_supported_uniform_pressure_kirchhoff(
    *,
    length_x: float,
    length_y: float,
    bending_stiffness: float,
    pressure: float,
    modes: int = 3,
    x: float | None = None,
    y: float | None = None,
) -> float:
    """Independent Kirchhoff limit using the same odd Navier truncation."""
    lx, ly, d = float(length_x), float(length_y), float(bending_stiffness)
    x_eval = 0.5 * lx if x is None else float(x)
    y_eval = 0.5 * ly if y is None else float(y)
    value = 0.0
    for m in range(1, 2 * int(modes), 2):
        alpha = m * np.pi / lx
        for n in range(1, 2 * int(modes), 2):
            beta = n * np.pi / ly
            q_mn = 16.0 * float(pressure) / (m * n * np.pi**2)
            value += (
                q_mn
                / (d * (alpha * alpha + beta * beta) ** 2)
                * np.sin(alpha * x_eval)
                * np.sin(beta * y_eval)
            )
    return float(value)


__all__ = [
    "simply_supported_uniform_pressure_fsdt",
    "simply_supported_uniform_pressure_kirchhoff",
]
