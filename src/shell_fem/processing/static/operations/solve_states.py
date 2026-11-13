"""System states passed between the static solve stages.

Each stage takes one state and returns the next:

    mesh --AssembleGlobalSystem--> GlobalSystem
         --ConstrainGlobalSystem--> ConstrainedSystem      U = T q + g
         --ReduceConstrainedSystem--> ReducedSystem        K_r = T^T K T
         --SolveReducedSystem--> FactorisedSystem          (one factorisation, many loads)
    F    --SolveReducedSystem.solve--> q                   K_r q = T^T (F - K g)
    q    --ReconstructGlobalField--> U
    U    --ComputePrimaryResults--> SolutionResult         R = K U - F
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class GlobalSystem:
    """Assembled global stiffness K (springs included), before any constraint."""

    K: object  # dense ndarray or SciPy sparse matrix
    n_dof: int


@dataclass(frozen=True)
class ConstrainedSystem:
    """The global system plus the constraint map U = T q + g.

    Prescribed DOFs drop out of q (their values sit in g); each slave DOF of a
    linear constraint is written through its masters in the rows of T.
    """

    K: object
    T: object
    g: np.ndarray

    @property
    def n_dof(self) -> int:
        return int(self.T.shape[0])

    @property
    def n_free(self) -> int:
        return int(self.T.shape[1])


@dataclass(frozen=True)
class ReducedSystem:
    """K_r = T^T K T and the load offset K g of the prescribed values."""

    constrained: ConstrainedSystem
    K_r: object
    K_g: np.ndarray

    def reduce_load(self, F: np.ndarray) -> np.ndarray:
        """F_r = T^T (F - K g)."""
        return np.asarray(self.constrained.T.T @ (F - self.K_g)).ravel()


@dataclass(frozen=True)
class FactorisedSystem:
    """A reduced system with its factorisation, ready for any number of load vectors."""

    reduced: ReducedSystem
    solve: Callable[[np.ndarray], np.ndarray]
    solver_name: str
