"""System states passed between the linear buckling stages.

The prestress comes from a linear static solve of the reference load; the buckling
stages then run on the same constraints:

    SolutionResult --AssembleGeometricStiffness--> GeometricSystem    K_g(N of the reference load)
    ReducedSystem + GeometricSystem --SolveBucklingEigenproblem--> BucklingEigenState
                                                   K_r phi = lambda (-T^T K_g T) phi
    BucklingEigenState --ComputeBucklingModes--> LinearBucklingResults   modes U = T phi

lambda is the factor on the reference load at which the structure buckles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class GeometricSystem:
    """Global geometric stiffness K_g of the reference prestress, before any constraint."""

    K_g: object  # dense ndarray or SciPy sparse matrix
    n_dof: int


@dataclass(frozen=True)
class BucklingEigenState:
    """Load factors (ascending) and the matching modes on the independent DOFs (columns of ``modes``)."""

    load_factors: np.ndarray
    modes: np.ndarray
    method: str
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def n_modes(self) -> int:
        return int(self.load_factors.size)
