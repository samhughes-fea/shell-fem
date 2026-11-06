"""Linear static simulation runner.

The runner sequences the static stages in ``shell_fem.processing.static`` and owns
their results; it does no mechanics itself:

    AssembleGlobalSystem -> ConstrainGlobalSystem -> ReduceConstrainedSystem
    -> SolveReducedSystem (factorise once)
    then for each load case:
    AssembleLoadVector -> SolveReducedSystem.solve -> ReconstructGlobalField -> ComputePrimaryResults

Load combinations are superposed exactly from the basic cases. Wall time of every
stage is recorded in ``LinearStaticResults.diagnostics``.

``StructuralGlobalSolver`` is the earlier interface (``solve``, ``solve_many``,
``solve_combinations``) and runs the same stages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.pre_processing.load_library.load_case import LoadCase, LoadCombination
from shell_fem.processing.static.operations.assembly import AssembleGlobalSystem, AssembleLoadVector, GlobalAssembler
from shell_fem.processing.static.operations.constraints import ConstrainGlobalSystem, ConstraintEliminator
from shell_fem.processing.static.operations.linear_solvers import LinearSystemSolver, default_linear_solver
from shell_fem.processing.static.operations.reconstruction import ReconstructGlobalField
from shell_fem.processing.static.operations.reduction import ReduceConstrainedSystem
from shell_fem.processing.static.operations.solution import SolveReducedSystem
from shell_fem.processing.static.results.compute_primary import ComputePrimaryResults, combine_solutions
from shell_fem.processing.static.results.containers import SolutionResult


@dataclass
class StaticDiagnostics:
    """Wall time (seconds) of each stage, summed over load cases, and the system size."""

    phase_times: Dict[str, float] = field(default_factory=dict)
    n_dof: int = 0
    n_free: int = 0
    linear_solver: str = ""

    def add(self, phase: str, seconds: float) -> None:
        self.phase_times[phase] = self.phase_times.get(phase, 0.0) + seconds

    @property
    def total_time(self) -> float:
        return sum(self.phase_times.values())


@dataclass
class LinearStaticResults:
    """Solutions by load case (and combination) name, with the solve diagnostics."""

    cases: Dict[str, SolutionResult]
    diagnostics: StaticDiagnostics

    def __getitem__(self, name: str) -> SolutionResult:
        return self.cases[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)


class _Timer:
    def __init__(self, diagnostics: StaticDiagnostics, phase: str):
        self.diagnostics, self.phase = diagnostics, phase

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        self.diagnostics.add(self.phase, time.perf_counter() - self.t0)


def _run_stages(mesh: FEAMesh, bcs: BoundaryConditions, load_cases: Sequence[LoadCase], assembler: GlobalAssembler,
                eliminator: ConstraintEliminator, linear_solver: LinearSystemSolver,
                diagnostics: StaticDiagnostics) -> List[SolutionResult]:
    """Run the static stages for several load cases on one mesh and one set of constraints."""
    start = time.perf_counter()
    with _Timer(diagnostics, "assemble"):
        system = AssembleGlobalSystem(mesh, bcs, assembler).run()
    with _Timer(diagnostics, "constrain"):
        constrained = ConstrainGlobalSystem(mesh, bcs, eliminator).run(system)
    with _Timer(diagnostics, "reduce"):
        reduced = ReduceConstrainedSystem().run(constrained)
    solver = SolveReducedSystem(linear_solver)
    with _Timer(diagnostics, "factorise"):
        factorised = solver.factorise(reduced)
    setup = time.perf_counter() - start
    diagnostics.n_dof, diagnostics.n_free, diagnostics.linear_solver = mesh.n_dof, constrained.n_free, linear_solver.name

    loads, reconstruct, primary = AssembleLoadVector(mesh, assembler), ReconstructGlobalField(), \
        ComputePrimaryResults(mesh, system)
    info = {"n_dof": mesh.n_dof, "n_free": constrained.n_free, "linear_solver": linear_solver.name}
    results: List[SolutionResult] = []
    for load_case in load_cases:
        t0 = time.perf_counter()
        with _Timer(diagnostics, "assemble_loads"):
            F = loads.run(load_case)
        with _Timer(diagnostics, "solve"):
            q = solver.solve(factorised, F)
        with _Timer(diagnostics, "reconstruct"):
            U = reconstruct.run(reduced, q)
        with _Timer(diagnostics, "primary_results"):
            elapsed = time.perf_counter() - t0 + (setup if not results else 0.0)
            results.append(primary.run(U, F, load_case, elapsed, info))
    return results


class LinearStaticRunner:
    """Linear static analysis of one mesh under one set of boundary conditions.

    >>> results = LinearStaticRunner(mesh, bcs, [dead, wind], [LoadCombination("ULS", {...})]).execute().get_results()
    >>> results["ULS"].displacements, results.diagnostics.phase_times

    One assembly and one factorisation serve every load case; combinations are exact
    superpositions of the basic cases.
    """

    def __init__(
        self,
        mesh: FEAMesh,
        bcs: BoundaryConditions,
        load_cases: Sequence[LoadCase],
        combinations: Sequence[LoadCombination] = (),
        linear_solver: Optional[LinearSystemSolver] = None,
        assembler: Optional[GlobalAssembler] = None,
        eliminator: Optional[ConstraintEliminator] = None,
    ):
        if isinstance(load_cases, LoadCase):
            load_cases = [load_cases]
        names = [c.name for c in load_cases]
        if len(set(names)) != len(names):
            raise ValueError("Load case names must be unique")
        self.mesh, self.bcs = mesh, bcs
        self.load_cases, self.combinations = list(load_cases), list(combinations)
        self.linear_solver = linear_solver or default_linear_solver()
        self.assembler = assembler or GlobalAssembler()
        self.eliminator = eliminator or ConstraintEliminator()
        self._results: Optional[LinearStaticResults] = None

    def execute(self) -> "LinearStaticRunner":
        diagnostics = StaticDiagnostics()
        basic = _run_stages(self.mesh, self.bcs, self.load_cases, self.assembler, self.eliminator,
                            self.linear_solver, diagnostics)
        cases: Dict[str, SolutionResult] = {c.name: r for c, r in zip(self.load_cases, basic)}
        by_name = {c.name: c for c in self.load_cases}
        with _Timer(diagnostics, "combinations"):
            for combo in self.combinations:
                terms = [(cases[case.name], factor) for case, factor in combo.terms(by_name)]
                cases[combo.name] = combine_solutions(terms, name=combo.name)
        self._results = LinearStaticResults(cases, diagnostics)
        return self

    def get_results(self) -> LinearStaticResults:
        if self._results is None:
            raise RuntimeError("Call execute() before get_results()")
        return self._results


class StructuralGlobalSolver:
    """Assemble, constrain and solve [K]{U} = {F} for a mesh (runs the same stages as ``LinearStaticRunner``)."""

    def __init__(
        self,
        assembler: Optional[GlobalAssembler] = None,
        linear_solver: Optional[LinearSystemSolver] = None,
        eliminator: Optional[ConstraintEliminator] = None,
    ):
        self.assembler = assembler or GlobalAssembler()
        self.linear_solver = linear_solver or default_linear_solver()
        self.eliminator = eliminator or ConstraintEliminator()
        self.last_diagnostics: Optional[StaticDiagnostics] = None

    def solve(self, mesh: FEAMesh, bcs: BoundaryConditions, load_case: LoadCase) -> SolutionResult:
        return self.solve_many(mesh, bcs, [load_case])[0]

    def solve_many(self, mesh: FEAMesh, bcs: BoundaryConditions, load_cases: Sequence[LoadCase]) -> List[SolutionResult]:
        """Solve several load cases with one assembly and one factorisation."""
        self.last_diagnostics = StaticDiagnostics()
        return _run_stages(mesh, bcs, load_cases, self.assembler, self.eliminator, self.linear_solver,
                           self.last_diagnostics)

    def solve_combinations(
        self,
        mesh: FEAMesh,
        bcs: BoundaryConditions,
        load_cases: Sequence[LoadCase],
        combinations: Sequence[LoadCombination],
    ) -> Dict[str, SolutionResult]:
        """Solve the basic cases once and superpose them into each combination.

        Returns every basic case (by name) and every combination (by name).
        """
        runner = LinearStaticRunner(mesh, bcs, load_cases, combinations, self.linear_solver, self.assembler,
                                    self.eliminator).execute()
        self.last_diagnostics = runner.get_results().diagnostics
        return dict(runner.get_results().cases)
