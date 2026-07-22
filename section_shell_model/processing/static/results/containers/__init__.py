"""Results containers stratified by tier and resolution (mirror 1d-fem container_hopper)."""

from __future__ import annotations

from .container_hopper import (
    ShellHashinFiResult,
    ShellIndexMapSet,
    ShellPrimaryResultSet,
    ShellQuaternaryResultSet,
    ShellSecondaryResultSet,
    ShellTertiaryResultSet,
)
from .elemental_results import ShellElementalResults
from .gaussian_results import ShellGaussianResults
from .global_results import ShellGlobalResults
from .map_results import ShellMapEntry
from .nodal_results import ShellNodalResults
from .ply_results import ShellPlyResults

__all__ = [
    "ShellElementalResults",
    "ShellGaussianResults",
    "ShellGlobalResults",
    "ShellHashinFiResult",
    "ShellIndexMapSet",
    "ShellMapEntry",
    "ShellNodalResults",
    "ShellPlyResults",
    "ShellPrimaryResultSet",
    "ShellQuaternaryResultSet",
    "ShellSecondaryResultSet",
    "ShellTertiaryResultSet",
]
