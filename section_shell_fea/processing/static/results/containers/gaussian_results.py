"""Gaussian-resolution results (secondary tier — 2×2 Gauss-point resultants)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShellGaussianResults:
    """Gauss-point shell resultants in **E** (``N11…M12`` = \\(N_{11}\\ldots\\)).

    Populated by
    :func:`processing.results.compute_secondary.compute_gaussian_fields.compute_gaussian_resultants`
    when the secondary orchestrator is called with ``compute_gaussian=True``.
    Parent ``(ξ, η)`` coordinates are element natural coordinates, not B/S/E axes.
    """

    fields: dict[str, Any] = field(default_factory=dict)


__all__ = ["ShellGaussianResults"]
