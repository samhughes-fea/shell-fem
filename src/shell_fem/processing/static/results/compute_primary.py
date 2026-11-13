"""Primary results: reactions from the solved displacements, and exact superposition."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.pre_processing.load_library.load_case import LoadCase
from shell_fem.processing.static.operations.solve_states import GlobalSystem
from shell_fem.processing.static.results.containers import SolutionResult


class ComputePrimaryResults:
    """Stage: reactions R = K U - F and the ``SolutionResult`` of one load case."""

    def __init__(self, mesh: FEAMesh, system: GlobalSystem):
        self.mesh = mesh
        self.system = system

    def run(self, U: np.ndarray, F: np.ndarray, load_case: Optional[LoadCase] = None, solve_time: float = 0.0,
            info: Optional[Dict[str, object]] = None) -> SolutionResult:
        R = np.asarray(self.system.K @ U).ravel() - F
        return SolutionResult(mesh=self.mesh, displacements=U, reactions=R, load_vector=F, solve_time=solve_time,
                              load_case=load_case, info=dict(info or {}))


def combine_solutions(
    terms: Sequence[Tuple[SolutionResult, float]], load_case: Optional[LoadCase] = None, name: str = "combination"
) -> SolutionResult:
    """Exact linear superposition of solutions on the same mesh and constraints.

    Prescribed non-zero displacements are scaled with their solution, so a combination
    of cases that each prescribe a settlement scales the settlement too.
    """
    if not terms:
        raise ValueError("combine_solutions needs at least one term")
    mesh = terms[0][0].mesh
    U = sum(f * s.displacements for s, f in terms)
    R = sum(f * s.reactions for s, f in terms)
    F = sum(f * s.load_vector for s, f in terms)
    if load_case is None and all(s.load_case is not None for s, _ in terms):
        load_case = LoadCase.combine([(s.load_case, f) for s, f in terms], name)
    return SolutionResult(mesh, U, R, F, 0.0, load_case, {"combination": name, "terms": len(terms)})
