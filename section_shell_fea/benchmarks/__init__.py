"""Performance benchmarks for section_shell_fea and orchestration call paths.

Heavy imports are lazy so diagnostic scripts (e.g. ``run_ccx_*_diag``) can be
executed via ``python -m ...`` without pulling optional optimiser dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "assess_deferred_perf_items",
    "benchmark_shell_fi_parallelism",
    "profile_station_topology_cache_churn",
    "write_benchmark_manifest",
]

def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import benchmark_section_solve_parallelism as _mod

        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
