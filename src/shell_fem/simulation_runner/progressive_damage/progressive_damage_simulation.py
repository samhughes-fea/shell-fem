"""Progressive damage runner: the incremental load-degrade-resolve loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from laminate_constitutive_model.laminate_failure_criteria import FailureCriterion, FailureMode, FailureResult, HashinCriterion
from laminate_constitutive_model.laminate_mechanics import Laminate
from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.pre_processing.load_library.load_case import LoadCase
from shell_fem.processing.static.results.containers import SolutionResult
from shell_fem.simulation_runner.static.linear_static_simulation import StructuralGlobalSolver
from shell_fem.processing.static.results.stress_recovery import ExactStressRecoverer
from shell_fem.processing.progressive_damage.degradation import DamageState, DegradationStrategy, SmearedCrackDegradation


@dataclass
class DamageStepRecord:
    load_factor: float
    iterations: int
    newly_failed: int
    total_damaged_plies: int
    max_failure_index: float
    max_displacement: float
    damage_events: List[Tuple[int, int, Tuple[str, ...]]] = field(default_factory=list)


@dataclass
class ProgressiveDamageResult:
    steps: List[DamageStepRecord]
    damage: Dict[Tuple[int, int], DamageState]
    final_solution: Optional[SolutionResult]
    mesh: FEAMesh
    first_ply_failure_factor: Optional[float]

    @property
    def damaged_plies(self) -> int:
        return sum(1 for d in self.damage.values() if d.any)


class ProgressiveDamageAnalysis:
    """Load-stepped analysis with equilibrium re-solves after each new damage event.

    At every load factor the model is solved, ply stresses are recovered at the
    element centre (bottom, middle and top of each ply), the failure criterion
    is evaluated, newly failed plies are degraded and the step is re-solved until
    no new damage appears or ``max_iterations`` is reached.
    """

    def __init__(
        self,
        criterion: Optional[FailureCriterion] = None,
        degradation: Optional[DegradationStrategy] = None,
        solver: Optional[StructuralGlobalSolver] = None,
        recoverer: Optional[ExactStressRecoverer] = None,
        max_iterations: int = 20,
        points_per_ply: int = 3,
    ):
        self.criterion = criterion or HashinCriterion("3d")
        self.degradation = degradation or SmearedCrackDegradation()
        self.solver = solver or StructuralGlobalSolver()
        self.recoverer = recoverer or ExactStressRecoverer()
        self.max_iterations = int(max_iterations)
        self.points_per_ply = int(points_per_ply)

    def _damage_from(self, result: FailureResult, stress: np.ndarray) -> DamageState:
        if result.mode is FailureMode.INTERACTIVE:
            return DamageState.from_interactive(result, stress)
        return DamageState.from_failure(result)

    def _evaluate(self, mesh: FEAMesh, solution: SolutionResult, load_case: LoadCase, virgin: Dict[int, Laminate]):
        """Return {(element, ply): DamageState} for failed plies and the max failure index."""
        found: Dict[Tuple[int, int], DamageState] = {}
        max_index = 0.0
        for element in mesh.elements:
            u_e = solution.element_displacements(element.element_id)
            state = load_case.state_for(element.element_id)
            if getattr(element, "IS_SOLID", False):
                profile = element.through_thickness(u_e, points_per_ply=self.points_per_ply, state=state)
                stresses = [p.stress_material for p in profile]
            else:
                profile = self.recoverer.through_thickness_profile(element, u_e, state=state, points_per_ply=self.points_per_ply)
                stresses = [p.material_stress_3d() for p in profile]
            for point, stress in zip(profile, stresses):
                strength = virgin[element.element_id].plies[point.ply_index].material.strength
                result = self.criterion.evaluate(stress, strength)
                max_index = max(max_index, result.index)
                damage = self._damage_from(result, stress)
                if damage.any:
                    key = (element.element_id, point.ply_index)
                    found[key] = found.get(key, DamageState()).merge(damage)
        return found, max_index

    def _apply(self, mesh: FEAMesh, virgin: Dict[int, Laminate], damage: Dict[Tuple[int, int], DamageState], keys) -> None:
        by_element: Dict[int, List[int]] = {}
        for e, k in keys:
            by_element.setdefault(e, []).append(k)
        for e in by_element:
            laminate = mesh.element(e).laminate
            for k in range(laminate.n_plies):
                state = damage.get((e, k))
                if state is None or not state.any:
                    continue
                virgin_ply = virgin[e].plies[k]
                elastic = self.degradation.degrade(virgin_ply.material.elastic, state)
                material = virgin_ply.material.with_elastic(elastic, suffix="+dmg")
                laminate = laminate.replace_ply(k, virgin_ply.with_material(material))
            mesh.set_element_laminate(e, laminate)

    def run(
        self,
        mesh: FEAMesh,
        bcs: BoundaryConditions,
        reference_load: LoadCase,
        load_factors: Sequence[float],
        scale_hygrothermal: bool = False,
    ) -> ProgressiveDamageResult:
        """Run the analysis on a copy of ``mesh``; ``reference_load`` is scaled by each factor."""
        mesh = mesh.copy()
        virgin = {e.element_id: e.laminate for e in mesh.elements}
        damage: Dict[Tuple[int, int], DamageState] = {}
        steps: List[DamageStepRecord] = []
        solution: Optional[SolutionResult] = None
        first_failure: Optional[float] = None
        for factor in load_factors:
            load_case = reference_load.scaled(factor, scale_hygrothermal)
            iterations = 0
            newly_total = 0
            events: List[Tuple[int, int, Tuple[str, ...]]] = []
            max_index = 0.0
            while True:
                solution = self.solver.solve(mesh, bcs, load_case)
                iterations += 1
                found, max_index = self._evaluate(mesh, solution, load_case, virgin)
                changed = []
                for key, state in found.items():
                    previous = damage.get(key, DamageState())
                    merged = previous.merge(state)
                    if merged != previous:
                        damage[key] = merged
                        changed.append(key)
                        events.append((key[0], key[1], tuple(merged.labels())))
                if changed and first_failure is None:
                    first_failure = float(factor)
                newly_total += len(changed)
                if not changed or iterations >= self.max_iterations:
                    break
                self._apply(mesh, virgin, damage, changed)
            steps.append(
                DamageStepRecord(
                    load_factor=float(factor),
                    iterations=iterations,
                    newly_failed=newly_total,
                    total_damaged_plies=sum(1 for d in damage.values() if d.any),
                    max_failure_index=float(max_index),
                    max_displacement=float(np.max(np.abs(solution.displacement_array()[:, :3]))),
                    damage_events=events,
                )
            )
        return ProgressiveDamageResult(steps, damage, solution, mesh, first_failure)
