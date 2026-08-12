"""Results tier x resolution taxonomy and unwired-computer register."""

from __future__ import annotations

from .compute_primary import (
    ComputeReactionForce,
    ComputeResidual,
    ShellPrimaryResultsOrchestrator,
)
from .compute_quaternary import ShellQuaternaryResultsOrchestrator
from .compute_secondary import ShellSecondaryResultsOrchestrator
from .compute_tertiary import ShellTertiaryResultsOrchestrator

from .containers import (
    ShellElementalResults,
    ShellGlobalResults,
    ShellHashinFiResult,
    ShellIndexMapSet,
    ShellMapEntry,
    ShellPlyResults,
    ShellPrimaryResultSet,
    ShellQuaternaryResultSet,
    ShellSecondaryResultSet,
    ShellTertiaryResultSet,
)

__all__ = [
    "ComputeReactionForce",
    "ComputeResidual",
    "ShellElementalResults",
    "ShellGlobalResults",
    "ShellHashinFiResult",
    "ShellIndexMapSet",
    "ShellMapEntry",
    "ShellPlyResults",
    "ShellPrimaryResultSet",
    "ShellPrimaryResultsOrchestrator",
    "ShellQuaternaryResultSet",
    "ShellQuaternaryResultsOrchestrator",
    "ShellSecondaryResultSet",
    "ShellSecondaryResultsOrchestrator",
    "ShellTertiaryResultSet",
    "ShellTertiaryResultsOrchestrator",
]
