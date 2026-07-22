"""Primary result computers (reactions, residual, orchestrator)."""

from __future__ import annotations

from .compute_reaction_force import ComputeReactionForce
from .compute_residual import ComputeResidual
from .primary_results_orchestrator import ShellPrimaryResultsOrchestrator
from .section_deformation_average import (
    SectionDeformationAverage1a,
    compute_section_deformation_average_1a,
)

__all__ = [
    "ComputeReactionForce",
    "ComputeResidual",
    "SectionDeformationAverage1a",
    "ShellPrimaryResultsOrchestrator",
    "compute_section_deformation_average_1a",
]
