"""Theory / backend dispatch registries (not element kernels)."""

from __future__ import annotations

from .shell_solve_registry import BACKENDS, register_backend
from .station_theory_dispatch import (
    StationTheorySolveRequest,
    StationTheorySolveResult,
    dispatch_station_theory,
)

__all__ = [
    "BACKENDS",
    "StationTheorySolveRequest",
    "StationTheorySolveResult",
    "dispatch_station_theory",
    "register_backend",
]
