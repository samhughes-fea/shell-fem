"""FE solve-phase timing collector (worker-internal; no campaign file I/O).

Phase names match ``SectionLinearStaticRunner`` / ``build_formulation_cache``
methods. Totals freeze into :class:`DiagnosticsBin` on the cascade result.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from ...results.containers.diagnostics_bin import DiagnosticsBin

# Per-element formulation phase names + collector Protocol are owned by the neutral
# pre_processing seam so build_formulation_cache does not depend on this module.
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.phase_timing import (  # noqa: F401
    PHASE_ELEMENT_FORCE_VECTORS,
    PHASE_ELEMENT_STIFFNESS_MATRICES,
    PhaseCollector,
)

# Station constraint composition (execute / _prepare_station_system).
PHASE_COMPOSE_CONSTRAINTS = "compose_constraints"

# Kinematics (solve_linear_system_only / solve_constrained_system).
PHASE_PREPARE_LOCAL_SYSTEM = "prepare_local_system"
PHASE_ASSEMBLE_GLOBAL_SYSTEM = "assemble_global_system"
PHASE_CONSTRAIN_GLOBAL_SYSTEM = "constrain_global_system"
PHASE_REDUCE_CONSTRAINED_SYSTEM = "reduce_constrained_system"
PHASE_SOLVE_REDUCED_SYSTEM = "solve_reduced_system"
PHASE_RECONSTRUCT_GLOBAL_FIELD = "reconstruct_global_field"

# Cascade result tiers.
PHASE_PRIMARY_RESULTS = "primary_results"
PHASE_SECONDARY_RESULTS = "secondary_results"
PHASE_TERTIARY_RESULTS = "tertiary_results"
PHASE_QUATERNARY_RESULTS = "quaternary_results"


class SolvePhaseCollector:
    """Accumulates named FE phase wall times for one worker call."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._t0 = time.perf_counter()

    def reset(self) -> None:
        self._totals.clear()
        self._counts.clear()
        self._t0 = time.perf_counter()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._totals[name] = self._totals.get(name, 0.0) + elapsed
            self._counts[name] = self._counts.get(name, 0) + 1

    def freeze(self) -> DiagnosticsBin:
        return DiagnosticsBin(
            phase_totals_s=dict(self._totals),
            phase_counts=dict(self._counts),
            wall_s=time.perf_counter() - self._t0,
        )


class _NullPhaseCollector:
    """No-op collector for assembly helpers called outside a runner."""

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        del name
        yield

    def reset(self) -> None:
        return None

    def freeze(self) -> DiagnosticsBin:
        return DiagnosticsBin()


_NULL_PHASE_COLLECTOR = _NullPhaseCollector()


__all__ = [
    "PhaseCollector",
    "SolvePhaseCollector",
    "_NULL_PHASE_COLLECTOR",
    "PHASE_ELEMENT_STIFFNESS_MATRICES",
    "PHASE_ELEMENT_FORCE_VECTORS",
    "PHASE_COMPOSE_CONSTRAINTS",
    "PHASE_PREPARE_LOCAL_SYSTEM",
    "PHASE_ASSEMBLE_GLOBAL_SYSTEM",
    "PHASE_CONSTRAIN_GLOBAL_SYSTEM",
    "PHASE_REDUCE_CONSTRAINED_SYSTEM",
    "PHASE_SOLVE_REDUCED_SYSTEM",
    "PHASE_RECONSTRUCT_GLOBAL_FIELD",
    "PHASE_PRIMARY_RESULTS",
    "PHASE_SECONDARY_RESULTS",
    "PHASE_TERTIARY_RESULTS",
    "PHASE_QUATERNARY_RESULTS",
]
