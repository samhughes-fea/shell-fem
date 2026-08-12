"""Nodal-resolution results (secondary tier — extrapolation from Gaussian)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShellNodalResults:
    """Nodal shell resultants in **E**, extrapolated from 2×2 Gauss points.

    Populated by
    :func:`processing.results.compute_secondary.compute_nodal_fields.compute_nodal_resultants`
    when the secondary orchestrator is called with ``compute_nodal=True``.
    """

    fields: dict[str, Any] = field(default_factory=dict)


__all__ = ["ShellNodalResults"]
