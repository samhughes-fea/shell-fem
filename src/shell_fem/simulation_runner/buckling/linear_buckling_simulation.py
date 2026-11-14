"""Linear buckling simulation runner.

The runner sequences the static and buckling stages and owns their results; it
does no mechanics itself:

    AssembleGlobalSystem -> ConstrainGlobalSystem -> ReduceConstrainedSystem
    -> SolveReducedSystem (factorise once)
    -> AssembleLoadVector -> SolveReducedSystem.solve -> ReconstructGlobalField
    -> ComputePrimaryResults                      (the reference static solution)
    -> AssembleGeometricStiffness                 (K_g of its membrane prestress)
    -> SolveBucklingEigenproblem                  (K_r phi = lambda (-K_g,r) phi)
    -> ComputeBucklingModes

The load factor of a mode multiplies the whole reference load case. Loads that
should not scale (a dead weight under a varying wind load, say) need a nonlinear
or a two-step analysis, which this runner does not do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.pre_processing.load_library.load_case import LoadCase
from shell_fem.processing.buckling.operations.eigen import SolveBucklingEigenproblem
from shell_fem.processing.buckling.operations.geometric_stiffness import AssembleGeometricStiffness
from shell_fem.processing.buckling.results.modes import BucklingMode, ComputeBucklingModes
from shell_fem.processing.static.operations.assembly import AssembleGlobalSystem, AssembleLoadVector, GlobalAssembler
from shell_fem.processing.static.operations.constraints import ConstrainGlobalSystem, ConstraintEliminator
from shell_fem.processing.static.operations.linear_solvers import LinearSystemSolver, default_linear_solver
from shell_fem.processing.static.operations.reconstruction import ReconstructGlobalField
from shell_fem.processing.static.operations.reduction import ReduceConstrainedSystem
from shell_fem.processing.static.operations.solution import SolveReducedSystem
from shell_fem.processing.static.results.compute_primary import ComputePrimaryResults
from shell_fem.processing.static.results.containers import SolutionResult
from shell_fem.simulation_runner.static.linear_static_simulation import StaticDiagnostics, _Timer


@dataclass
class LinearBucklingResults:
    """The buckling modes (lowest load factor first), the reference static solution and the diagnostics."""

    modes: List[BucklingMode]
    reference: SolutionResult
    diagnostics: StaticDiagnostics
    eigen_method: str = ""

    @property
    def load_factors(self) -> np.ndarray:
        return np.array([m.load_factor for m in self.modes])

    @property
    def critical_load_factor(self) -> float:
        """The lowest load factor (nan when the reference load compresses nothing)."""
        return self.modes[0].load_factor if self.modes else float("nan")

    def __getitem__(self, k: int) -> BucklingMode:
        return self.modes[k]

    def __iter__(self) -> Iterator[BucklingMode]:
        return iter(self.modes)

    def __len__(self) -> int:
        return len(self.modes)


class LinearBucklingRunner:
    """Linear (eigenvalue) buckling of one mesh under one reference load case.

    >>> results = LinearBucklingRunner(mesh, bcs, load, n_modes=4).execute().get_results()
    >>> results.critical_load_factor, results[0].displacement_array()

    The prestress is the membrane field of the linear static solution of ``load_case``;
    the buckling problem uses the same constraints. Shell elements only.
    """

    def __init__(
        self,
        mesh: FEAMesh,
        bcs: BoundaryConditions,
        load_case: LoadCase,
        n_modes: int = 5,
        eigensolver: Optional[SolveBucklingEigenproblem] = None,
        linear_solver: Optional[LinearSystemSolver] = None,
        assembler: Optional[GlobalAssembler] = None,
        eliminator: Optional[ConstraintEliminator] = None,
    ):
        self.mesh, self.bcs, self.load_case = mesh, bcs, load_case
        self.eigensolver = eigensolver or SolveBucklingEigenproblem(n_modes)
        self.linear_solver = linear_solver or default_linear_solver()
        self.assembler = assembler or GlobalAssembler()
        self.eliminator = eliminator or ConstraintEliminator()
        self._results: Optional[LinearBucklingResults] = None

    def execute(self) -> "LinearBucklingRunner":
        mesh, bcs, d = self.mesh, self.bcs, StaticDiagnostics()
        with _Timer(d, "assemble"):
            system = AssembleGlobalSystem(mesh, bcs, self.assembler).run()
        with _Timer(d, "constrain"):
            constrained = ConstrainGlobalSystem(mesh, bcs, self.eliminator).run(system)
        with _Timer(d, "reduce"):
            reduced = ReduceConstrainedSystem().run(constrained)
        solver = SolveReducedSystem(self.linear_solver)
        with _Timer(d, "factorise"):
            factorised = solver.factorise(reduced)
        with _Timer(d, "assemble_loads"):
            F = AssembleLoadVector(mesh, self.assembler).run(self.load_case)
        with _Timer(d, "solve"):
            U = ReconstructGlobalField().run(reduced, solver.solve(factorised, F))
        info = {"n_dof": mesh.n_dof, "n_free": constrained.n_free, "linear_solver": self.linear_solver.name}
        with _Timer(d, "primary_results"):
            reference = ComputePrimaryResults(mesh, system).run(U, F, self.load_case, 0.0, info)
        with _Timer(d, "assemble_geometric"):
            geometric = AssembleGeometricStiffness(mesh, self.assembler.sparse).run(reference)
        with _Timer(d, "eigen"):
            state = self.eigensolver.run(factorised, geometric)
        with _Timer(d, "modes"):
            modes = ComputeBucklingModes(mesh).run(reduced, state)
        d.n_dof, d.n_free, d.linear_solver = mesh.n_dof, constrained.n_free, self.linear_solver.name
        self._results = LinearBucklingResults(modes, reference, d, state.method)
        return self

    def get_results(self) -> LinearBucklingResults:
        if self._results is None:
            raise RuntimeError("Call execute() before get_results()")
        return self._results
