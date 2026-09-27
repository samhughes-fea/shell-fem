"""Warm-median microbenchmarks for static performance follow-on families.

Report-only — does not change physics. Run from repo root::

    python src/blade_structure_utilities/fem_utilities/section_shell_fea/examples/bench_static_performance_followon.py
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_primary.primary_results_orchestrator import (
    ShellPrimaryResultsOrchestrator,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_quaternary.compute_hashin_fi import (
    interface_shear_screening_from_tau,
    interface_shear_screening_from_tau_batch,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_tertiary.recovery_operators import (
    fsdt_interface_shear_profile,
    fsdt_interface_shear_profile_coefficients,
)


def _median_ms(fn, *, warmup: int = 3, repeats: int = 25) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples))


def bench_primary() -> None:
    n = 2000
    diag = np.full(n, 4.0)
    off = np.full(n - 1, -1.0)
    K = sp.diags([off, diag, off], offsets=[-1, 0, 1], format="csr")
    f = np.ones(n)
    d = np.linspace(0.0, 1.0, n)
    fixed = {0, n - 1}

    def run() -> None:
        ShellPrimaryResultsOrchestrator(K=K, f=f, d_global=d, fixed=fixed).compute()

    print(f"primary_orchestrator_ms\t{_median_ms(run):.3f}")


def bench_shear_coeffs() -> None:
    x3 = np.linspace(-0.02, 0.02, 17)
    h = 0.04
    g13 = np.linspace(1.0, 2.0, 16)
    g23 = np.linspace(1.2, 1.8, 16)
    q = [(float(a), float(b)) for a, b in zip(np.linspace(1, 20, 200), np.linspace(-5, 5, 200))]

    def scalar() -> None:
        for q1, q2 in q:
            fsdt_interface_shear_profile(
                q1=q1, q2=q2, x3_iface=x3, h=h, g13_ply=g13, g23_ply=g23,
            )

    def coeff_reuse() -> None:
        c13, c23 = fsdt_interface_shear_profile_coefficients(
            x3_iface=x3, h=h, g13_ply=g13, g23_ply=g23,
        )
        for q1, q2 in q:
            _ = q1 * c13
            _ = q2 * c23

    print(f"shear_profile_scalar_ms\t{_median_ms(scalar):.3f}")
    print(f"shear_profile_coeff_reuse_ms\t{_median_ms(coeff_reuse):.3f}")


class _Ply:
    def __init__(self, s13: float, s23: float) -> None:
        self.S13 = s13
        self.S23 = s23
        self.S12 = 0.0


class _Lam:
    def __init__(self) -> None:
        self.plies = [
            (_Ply(50e6, 40e6), 0.0),
            (_Ply(55e6, 42e6), 45.0),
            (_Ply(50e6, 40e6), -45.0),
            (_Ply(48e6, 38e6), 90.0),
        ]


def bench_interface_shear() -> None:
    lam = _Lam()
    rng = np.random.default_rng(1)
    tau = rng.normal(scale=1e5, size=(400, 5, 2))
    tau[:, 0, :] = 0.0
    tau[:, -1, :] = 0.0

    def scalar() -> None:
        for i in range(tau.shape[0]):
            interface_shear_screening_from_tau(tau[i], lam)

    def batch() -> None:
        interface_shear_screening_from_tau_batch(tau, lam)

    print(f"interface_shear_scalar_ms\t{_median_ms(scalar, repeats=15):.3f}")
    print(f"interface_shear_batch_ms\t{_median_ms(batch, repeats=15):.3f}")


def main() -> None:
    bench_primary()
    bench_shear_coeffs()
    bench_interface_shear()


if __name__ == "__main__":
    main()
