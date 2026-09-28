"""Neutral phase-timing seam (no ``processing`` dependency).

The formulation-cache builder times ``element_stiffness_matrices`` /
``element_force_vectors`` phases through a collector, but must not depend on the
``processing`` layer to do so (that would invert the pre_processing → processing
dependency). This module owns the phase-name constants, the collector
:class:`PhaseCollector` Protocol, and a null collector.

The concrete ``SolvePhaseCollector`` (which freezes into a processing-layer
``DiagnosticsBin``) still lives in
``processing.static.diagnostics.solve.solve_phase_timing`` and re-exports these
neutral names so the phase strings stay identical across layers.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Protocol

# Per-element formulation phases (build_formulation_cache). Phase strings are the
# single source of truth shared with the processing collector.
PHASE_ELEMENT_STIFFNESS_MATRICES = "element_stiffness_matrices"
PHASE_ELEMENT_FORCE_VECTORS = "element_force_vectors"


class PhaseCollector(Protocol):
    """Minimal collector seam consumed by ``build_formulation_cache``."""

    @contextmanager
    def phase(self, name: str) -> Iterator[None]: ...

    def reset(self) -> None: ...

    def freeze(self) -> Any: ...


class _NullPhaseCollector:
    """No-op collector for cache builds called outside a runner."""

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        del name
        yield

    def reset(self) -> None:
        return None

    def freeze(self) -> Any:
        return None


_NULL_PHASE_COLLECTOR = _NullPhaseCollector()


__all__ = [
    "PHASE_ELEMENT_FORCE_VECTORS",
    "PHASE_ELEMENT_STIFFNESS_MATRICES",
    "PhaseCollector",
    "_NULL_PHASE_COLLECTOR",
]
