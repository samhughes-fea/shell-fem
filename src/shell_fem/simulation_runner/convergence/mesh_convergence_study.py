"""Mesh convergence study: one analysis on a ladder of meshes, with the GCI of each result.

``MeshConvergenceStudy`` runs a user function on each mesh level, records the
representative element size h of its mesh (``processing.diagnostics.mesh_size``) and the
scalar results it returns, and estimates the discretisation error of each result with the
ASME grid convergence index (``processing.diagnostics.grid_convergence_index``) over the
three finest meshes, or over any three of them.

``blade_convergence_study`` is the full-blade case: a blade built at each level, the same
load cases on each, solved by ``BladeStaticRunner``, with the tip deflection and twist,
the root loads and, optionally, the linear buckling factor of one case tracked.

The runner sequences; it computes no mechanics itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from shell_fem.processing.diagnostics.gci import GCIResult, grid_convergence_index, mesh_size


@dataclass
class MeshLevel:
    """One mesh of the ladder: its level parameter, size h, DOF count, results and run time."""

    level: Hashable
    h: float
    n_dof: int
    values: Dict[str, float]
    seconds: float


@dataclass
class MeshConvergenceResults:
    """The levels of a study, finest last (smallest h last)."""

    levels: List[MeshLevel]
    safety_factor: float = 1.25
    atol: Dict[str, float] = field(default_factory=dict)
    _cache: Dict[Tuple[str, Tuple[int, ...]], GCIResult] = field(default_factory=dict, repr=False)

    @property
    def quantities(self) -> List[str]:
        return list(self.levels[0].values) if self.levels else []

    @property
    def h(self) -> np.ndarray:
        return np.array([lv.h for lv in self.levels])

    def values(self, name: str) -> np.ndarray:
        return np.array([lv.values[name] for lv in self.levels])

    def gci(self, name: str, levels: Optional[Sequence[int]] = None) -> GCIResult:
        """GCI of ``name`` over three levels (indices into ``levels``; the three finest by default)."""
        if len(self.levels) < 3:
            raise ValueError("a GCI needs at least three mesh levels")
        idx = tuple(int(i) for i in (levels if levels is not None else range(len(self.levels) - 3, len(self.levels))))
        key = (name, idx)
        if key not in self._cache:
            self._cache[key] = grid_convergence_index([self.levels[i].h for i in idx],
                                                      [self.levels[i].values[name] for i in idx], self.safety_factor,
                                                      atol=self.atol.get(name, 0.0))
        return self._cache[key]

    def reliable(self, name: str, order_tol: float = 0.3) -> bool:
        """Whether the GCI of ``name`` on the finest triple can be trusted.

        It must be in the asymptotic range and, with four or more levels, the observed
        order of the finest triple must be within ``order_tol`` (relative) of the next.
        """
        g = self.gci(name)
        if g.converged:
            return True
        if not g.in_asymptotic_range():
            return False
        if len(self.levels) < 4:
            return True
        n = len(self.levels)
        g2 = self.gci(name, range(n - 4, n - 1))
        if g2.converged:
            return True
        return g2.in_asymptotic_range() and abs(g2.order - g.order) <= order_tol * g.order

    def summary(self) -> Dict[str, GCIResult]:
        """GCI of every quantity over the three finest levels."""
        return {name: self.gci(name) for name in self.quantities}

    def table(self) -> str:
        """A plain-text table of the levels and the GCI of each quantity on the finest triple.

        A GCI marked * converges oscillatorily; one marked ? is not ``reliable``.
        """
        names = self.quantities
        lines = [f"{'h':>10}{'DOFs':>9}{'s':>8}" + "".join(f"{n:>16}" for n in names)]
        for lv in self.levels:
            lines.append(f"{lv.h:10.4g}{lv.n_dof:9d}{lv.seconds:8.1f}" + "".join(f"{lv.values[n]:16.6g}" for n in names))
        if len(self.levels) >= 3:
            rows = {"p": [], "extrapolated": [], "GCI fine (%)": []}
            for n in names:
                g = self.gci(n)
                rows["p"].append(f"{g.order:16.3g}")
                rows["extrapolated"].append(f"{g.extrapolated:16.6g}")
                mark = ("*" if g.oscillatory else "") + ("" if self.reliable(n) else "?")
                rows["GCI fine (%)"].append(f"{100 * g.gci:{16 - len(mark)}.3g}{mark}")
            for label, cells in rows.items():
                lines.append(f"{label:>27}" + "".join(cells))
        return "\n".join(lines)

    def __iter__(self) -> Iterator[MeshLevel]:
        return iter(self.levels)

    def __len__(self) -> int:
        return len(self.levels)


class MeshConvergenceStudy:
    """Run ``analyse(level) -> (mesh, {name: value})`` on each level and estimate the GCI of each value.

    >>> study = MeshConvergenceStudy([8, 12, 18], lambda n: (mesh_of(n), {"tip": tip_of(n)}))
    >>> results = study.execute().get_results()
    >>> results.gci("tip").gci, results.gci("tip").extrapolated

    ``size`` maps a mesh to its representative element size (``mesh_size`` by default).
    Levels are sorted coarse to fine by that size. Every level must return the same names.
    ``atol`` gives, per name, a change between meshes small enough to count as none (for a
    result that is zero by symmetry, say).
    """

    def __init__(self, levels: Sequence[Hashable], analyse: Callable[[Hashable], Tuple[object, Dict[str, float]]],
                 size: Callable[[object], float] = mesh_size, safety_factor: float = 1.25,
                 atol: Optional[Dict[str, float]] = None):
        if len(levels) < 3:
            raise ValueError("a mesh convergence study needs at least three levels")
        self.levels = list(levels)
        self.analyse = analyse
        self.size = size
        self.safety_factor = float(safety_factor)
        self.atol = dict(atol or {})
        self._results: Optional[MeshConvergenceResults] = None

    def execute(self) -> "MeshConvergenceStudy":
        done: List[MeshLevel] = []
        for level in self.levels:
            start = time.perf_counter()
            mesh, values = self.analyse(level)
            seconds = time.perf_counter() - start
            values = {str(k): float(v) for k, v in values.items()}
            if done and set(values) != set(done[0].values):
                raise ValueError(f"level {level!r} returned {sorted(values)}, not {sorted(done[0].values)}")
            done.append(MeshLevel(level, float(self.size(mesh)), int(mesh.n_dof), values, seconds))
        done.sort(key=lambda lv: -lv.h)
        if any(a.h <= b.h for a, b in zip(done, done[1:])):
            raise ValueError("two levels have the same mesh size")
        self._results = MeshConvergenceResults(done, self.safety_factor, self.atol)
        return self

    def get_results(self) -> MeshConvergenceResults:
        if self._results is None:
            raise RuntimeError("Call execute() before get_results()")
        return self._results


def _root(i: int) -> Callable:
    return lambda case: float(case.root_load[i])


BLADE_QUANTITIES: Dict[str, Callable] = {
    "tip_flap": lambda case: float(case.tip_translation[1]),
    "tip_edge": lambda case: float(case.tip_translation[0]),
    "tip_twist": lambda case: float(case.tip_twist),
    "root_Mx": _root(3),
    "root_My": _root(4),
    "root_Mz": _root(5),
}
"""Results of a ``BladeCaseResult`` tracked by ``blade_convergence_study`` by default (SI units, radians)."""


def blade_convergence_study(build_blade: Callable[[Hashable], object], load_cases: Callable[[object], Sequence],
                            case: str, levels: Sequence[Hashable], quantities: Optional[Dict[str, Callable]] = None,
                            combinations: Sequence = (), buckling_modes: int = 0,
                            atol: Optional[Dict[str, float]] = None, **runner_options) -> MeshConvergenceStudy:
    """A ``MeshConvergenceStudy`` of a full blade under ``BladeStaticRunner``.

    ``build_blade(level)`` returns a ``BladeMesh``; ``load_cases(blade)`` the load cases on
    it (built per mesh, since spanwise loads are spread over each mesh's own bays). The
    results of ``case`` (a load case or combination name) are tracked: ``quantities``
    maps names to functions of its ``BladeCaseResult`` (``BLADE_QUANTITIES`` by default),
    and ``buckling_modes`` > 0 adds its critical buckling load factor as ``buckling``.
    """
    from shell_fem.simulation_runner.blade.blade_static_simulation import BladeStaticRunner

    tracked = dict(BLADE_QUANTITIES if quantities is None else quantities)

    def analyse(level):
        blade = build_blade(level)
        runner = BladeStaticRunner(blade, list(load_cases(blade)), list(combinations), buckling_modes=buckling_modes,
                                   buckling_cases=[case] if buckling_modes > 0 else None, **runner_options)
        result = runner.execute().get_results()[case]
        values = {name: fn(result) for name, fn in tracked.items()}
        if buckling_modes > 0:
            values["buckling"] = float(result.buckling.critical_load_factor)
        return blade.mesh, values

    return MeshConvergenceStudy(levels, analyse, atol=atol)


__all__ = ["BLADE_QUANTITIES", "MeshConvergenceResults", "MeshConvergenceStudy", "MeshLevel",
           "blade_convergence_study"]
