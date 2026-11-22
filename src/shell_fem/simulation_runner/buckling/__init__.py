"""Linear buckling runners.

``LinearBucklingRunner`` sequences the static stages and the buckling stages (dense or
Lanczos eigen-solve, negative factors when asked). ``linear_buckling(mesh, bcs,
load_case, n_modes=1)`` runs it and returns the positive load factors of one reference
load with full-system modes and per-element mode energy.
"""

from shell_fem.simulation_runner.buckling.buckling_simulation import BucklingResult, linear_buckling
from shell_fem.simulation_runner.buckling.linear_buckling_simulation import LinearBucklingResults, LinearBucklingRunner

__all__ = ["BucklingResult", "LinearBucklingResults", "LinearBucklingRunner", "linear_buckling"]
