"""Full-blade static runner: a blade clamped at the root under load cases given along the span.

``BladeStaticRunner`` runs ``LinearStaticRunner`` on the whole blade and, for every
load case and combination, collects what a blade check reads first:

- the root loads, the six resultants of the support reactions about a root point;
- the applied loads about the same point, and their sum with the root loads (the
  equilibrium residual, zero to round-off for a sound model);
- the deflection and twist of each span section (``deflection_profile``);
- optionally the linear buckling load factors and modes of each case.

The supports are a rigid root (``clamp_root``) unless other boundary conditions are
given. The runner does no mechanics itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence

import numpy as np

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.load_library.blade_loads import clamp_root
from shell_fem.pre_processing.load_library.load_case import LoadCase, LoadCombination
from shell_fem.processing.static.operations.linear_solvers import LinearSystemSolver
from shell_fem.processing.static.results.containers import SolutionResult
from shell_fem.processing.static.results.deflection import DeflectionProfile, deflection_profile
from shell_fem.simulation_runner.buckling.linear_buckling_simulation import LinearBucklingResults, LinearBucklingRunner
from shell_fem.simulation_runner.static.linear_static_simulation import LinearStaticRunner, StaticDiagnostics


def resultant_about(mesh, vector: np.ndarray, point: Sequence[float]) -> np.ndarray:
    """[Fx, Fy, Fz, Mx, My, Mz] of a nodal force vector (global DOFs) about ``point``."""
    nodal = mesh.standard_array(np.asarray(vector, dtype=float))
    r = mesh.coordinates_array() - np.asarray(point, dtype=float)
    force = nodal[:, :3].sum(axis=0)
    moment = np.cross(r, nodal[:, :3]).sum(axis=0) + nodal[:, 3:6].sum(axis=0)
    return np.concatenate([force, moment])


@dataclass
class BladeCaseResult:
    """One load case or combination on the blade.

    ``root_load`` is the resultant of the support reactions about ``BladeStaticResults.root_point``
    (what the hub applies to the blade); ``applied_load`` is the resultant of the
    applied loads about the same point. Their sum is ``equilibrium_residual``.
    """

    solution: SolutionResult
    root_load: np.ndarray
    applied_load: np.ndarray
    deflection: DeflectionProfile
    buckling: Optional[LinearBucklingResults] = None

    @property
    def equilibrium_residual(self) -> np.ndarray:
        return self.root_load + self.applied_load

    @property
    def tip_translation(self) -> np.ndarray:
        """Mean translation of the outermost span section."""
        return self.deflection.translation[-1].copy()

    @property
    def tip_twist(self) -> float:
        """Rigid twist of the outermost span section about z (radians)."""
        return float(self.deflection.twist[-1])


@dataclass
class BladeStaticResults:
    """Results by load case and combination name, the root point and the static solve diagnostics."""

    cases: Dict[str, BladeCaseResult]
    root_point: np.ndarray
    diagnostics: StaticDiagnostics

    def __getitem__(self, name: str) -> BladeCaseResult:
        return self.cases[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)


class BladeStaticRunner:
    """Linear static analysis of a full blade, with root loads and spanwise deflection for each case.

    >>> results = BladeStaticRunner(blade, [gravity, flap], buckling_modes=3).execute().get_results()
    >>> results["flap"].root_load, results["flap"].tip_translation, results["flap"].buckling.critical_load_factor

    ``blade`` is a ``BladeMesh`` (anything with ``.mesh``, ``nodes("root")`` and,
    optionally, ``span_positions``). ``root_point`` defaults to the pitch axis at the
    root, (0, 0, z_root). ``buckling_modes`` > 0 also runs linear buckling of every case
    and combination named in ``buckling_cases`` (all of them by default); each one
    scales its own reference load.
    """

    def __init__(
        self,
        blade,
        load_cases: Sequence[LoadCase],
        combinations: Sequence[LoadCombination] = (),
        bcs: Optional[BoundaryConditions] = None,
        root_point: Optional[Sequence[float]] = None,
        buckling_modes: int = 0,
        buckling_cases: Optional[Sequence[str]] = None,
        linear_solver: Optional[LinearSystemSolver] = None,
    ):
        if isinstance(load_cases, LoadCase):
            load_cases = [load_cases]
        self.blade = blade
        self.load_cases, self.combinations = list(load_cases), list(combinations)
        self.bcs = bcs if bcs is not None else clamp_root(blade)
        mesh = blade.mesh
        if root_point is None:
            z_root = float(np.min([mesh.node_coordinates(n)[2] for n in blade.nodes("root")]))
            root_point = (0.0, 0.0, z_root)
        self.root_point = np.asarray(root_point, dtype=float)
        self.buckling_modes = int(buckling_modes)
        self.buckling_cases = None if buckling_cases is None else set(buckling_cases)
        known = {c.name for c in self.load_cases} | {c.name for c in self.combinations}
        unknown = sorted((self.buckling_cases or set()) - known)
        if unknown:
            raise KeyError(f"buckling_cases names unknown load cases or combinations {unknown}")
        self.linear_solver = linear_solver
        self._results: Optional[BladeStaticResults] = None

    def _stations(self) -> Optional[np.ndarray]:
        z = np.asarray(getattr(self.blade, "span_positions", ()), dtype=float)
        if z.size == 0:
            return None
        nodes_z = self.blade.mesh.coordinates_array()[:, 2]
        tol = 1e-9 * max(1.0, float(np.max(np.abs(nodes_z))))
        if all(np.any(np.abs(nodes_z - s) <= tol) for s in z):
            return z
        return None

    def execute(self) -> "BladeStaticRunner":
        mesh = self.blade.mesh
        static = LinearStaticRunner(mesh, self.bcs, self.load_cases, self.combinations,
                                    linear_solver=self.linear_solver).execute().get_results()
        stations = self._stations()
        by_name = {c.name: c for c in self.load_cases}
        for combo in self.combinations:
            by_name[combo.name] = LoadCase.combine(combo.terms(by_name), combo.name)
        cases: Dict[str, BladeCaseResult] = {}
        for name, solution in static.cases.items():
            root = resultant_about(mesh, solution.reactions, self.root_point)
            applied = resultant_about(mesh, solution.load_vector, self.root_point)
            profile = deflection_profile(solution, axis=2, stations=stations)
            buckling = None
            if self.buckling_modes > 0 and (self.buckling_cases is None or name in self.buckling_cases):
                buckling = LinearBucklingRunner(mesh, self.bcs, by_name[name], n_modes=self.buckling_modes,
                                                linear_solver=self.linear_solver).execute().get_results()
            cases[name] = BladeCaseResult(solution, root, applied, profile, buckling)
        self._results = BladeStaticResults(cases, self.root_point, static.diagnostics)
        return self

    def get_results(self) -> BladeStaticResults:
        if self._results is None:
            raise RuntimeError("Call execute() before get_results()")
        return self._results
