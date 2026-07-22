"""Per-panel topology summary diagnostics for :mod:`..pre_processing.mesh.topology`.

``TopologyV2Diagnostics`` lives on the mesh module (pre_processing owns the type).
This module re-exports it for callers that historically imported from
``processing.diagnostics``.
"""

from __future__ import annotations

from ....pre_processing.mesh.topology import TopologyV2Diagnostics

__all__ = ["TopologyV2Diagnostics"]
