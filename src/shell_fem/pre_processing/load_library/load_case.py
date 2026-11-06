"""Load cases and load combinations.

A ``LoadCase`` collects nodal loads, pressures (uniform, per element or as a field),
line loads, body loads and hygrothermal states. A ``LoadCombination`` names factored
sums of load cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from laminate_constitutive_model.laminate_hygrothermal import NULL_STATE, HygrothermalState

if TYPE_CHECKING:
    from shell_fem.pre_processing.element_library.shell import ShellElement


PressureLike = Union[float, Callable[[np.ndarray], float]]


@dataclass(frozen=True)
class LineLoad:
    """Force (and optional moment) per unit length on every element edge whose nodes are all listed."""

    node_ids: frozenset
    force_per_length: Union[Tuple[float, ...], Callable[[np.ndarray], Sequence[float]]]

    def scaled(self, factor: float) -> "LineLoad":
        q = self.force_per_length
        if callable(q):
            return LineLoad(self.node_ids, lambda xyz, q=q, f=factor: f * np.asarray(q(xyz), dtype=float))
        return LineLoad(self.node_ids, tuple(factor * v for v in q))


@dataclass(frozen=True)
class BodyLoad:
    """Uniform acceleration (gravity, inertia relief) applied as rho_A * a."""

    acceleration: Tuple[float, float, float]
    element_ids: Optional[frozenset] = None

    def applies_to(self, element_id: int) -> bool:
        return self.element_ids is None or element_id in self.element_ids

    def scaled(self, factor: float) -> "BodyLoad":
        return BodyLoad(tuple(factor * a for a in self.acceleration), self.element_ids)


def _sum_pressures(parts: List[PressureLike]) -> PressureLike:
    numeric = sum(p for p in parts if not callable(p))
    fields_ = [p for p in parts if callable(p)]
    if not fields_:
        return float(numeric)
    return lambda xyz: numeric + sum(float(f(xyz)) for f in fields_)


def _scale_pressure(p: PressureLike, factor: float) -> PressureLike:
    if callable(p):
        return lambda xyz, p=p, f=factor: f * float(p(xyz))
    return p * factor


class LoadCase:
    """Nodal loads, pressures, line loads, body loads and hygrothermal states for one analysis."""

    def __init__(self, name: str = "LC1"):
        self.name = name
        self.nodal_loads: Dict[Tuple[int, int], float] = {}
        self.global_pressure: PressureLike = 0.0
        self.element_pressures: Dict[int, PressureLike] = {}
        self.line_loads: List[LineLoad] = []
        self.body_loads: List[BodyLoad] = []
        self.hygrothermal: HygrothermalState = NULL_STATE
        self.element_hygrothermal: Dict[int, HygrothermalState] = {}

    def add_nodal_load(self, node_id: int, dof: int, value: float) -> "LoadCase":
        key = (int(node_id), int(dof))
        self.nodal_loads[key] = self.nodal_loads.get(key, 0.0) + float(value)
        return self

    def add_pressure(self, pressure: PressureLike, element_ids: Optional[Iterable[int]] = None) -> "LoadCase":
        """Pressure along each element's +normal: a number, or a function of the global position.

        Applies to all elements when ``element_ids`` is omitted.
        """
        if element_ids is None:
            self.global_pressure = _sum_pressures([self.global_pressure, pressure])
        else:
            for e in element_ids:
                e = int(e)
                self.element_pressures[e] = _sum_pressures([self.element_pressures.get(e, 0.0), pressure])
        return self

    def add_line_load(
        self, node_ids: Iterable[int], force_per_length: Union[Sequence[float], Callable[[np.ndarray], Sequence[float]]]
    ) -> "LoadCase":
        """Line load on the element edges along ``node_ids`` (global [fx, fy, fz] or with [mx, my, mz])."""
        q = force_per_length if callable(force_per_length) else tuple(float(v) for v in force_per_length)
        if not callable(q) and len(q) not in (3, 6):
            raise ValueError("force_per_length needs 3 or 6 components")
        self.line_loads.append(LineLoad(frozenset(int(n) for n in node_ids), q))
        return self

    def add_body_load(self, acceleration: Sequence[float], element_ids: Optional[Iterable[int]] = None) -> "LoadCase":
        """Uniform acceleration, e.g. ``(0, 0, -9.81)`` for gravity in SI units."""
        a = tuple(float(v) for v in acceleration)
        if len(a) != 3:
            raise ValueError("acceleration needs 3 components")
        ids = None if element_ids is None else frozenset(int(e) for e in element_ids)
        self.body_loads.append(BodyLoad(a, ids))
        return self

    def set_hygrothermal(self, state: HygrothermalState, element_ids: Optional[Iterable[int]] = None) -> "LoadCase":
        if element_ids is None:
            self.hygrothermal = state
        else:
            for e in element_ids:
                self.element_hygrothermal[int(e)] = state
        return self

    def pressure_on(self, element_id: int) -> PressureLike:
        """Total pressure on an element: a float, or a function of position for pressure fields."""
        return _sum_pressures([self.global_pressure, self.element_pressures.get(element_id, 0.0)])

    def pressure_value(self, element: ShellElement) -> float:
        """Pressure on an element as one number (fields are evaluated at the centroid)."""
        p = self.pressure_on(element.element_id)
        return float(p(element.centroid)) if callable(p) else float(p)

    def has_pressure(self, element_id: int) -> bool:
        p = self.pressure_on(element_id)
        return callable(p) or p != 0.0

    def state_for(self, element_id: int) -> HygrothermalState:
        return self.element_hygrothermal.get(element_id, self.hygrothermal)

    def scaled(self, factor: float, scale_hygrothermal: bool = True) -> "LoadCase":
        new = LoadCase(f"{self.name}x{factor:g}")
        new.nodal_loads = {k: v * factor for k, v in self.nodal_loads.items()}
        new.global_pressure = _scale_pressure(self.global_pressure, factor)
        new.element_pressures = {k: _scale_pressure(v, factor) for k, v in self.element_pressures.items()}
        new.line_loads = [load.scaled(factor) for load in self.line_loads]
        new.body_loads = [load.scaled(factor) for load in self.body_loads]
        f = factor if scale_hygrothermal else 1.0
        new.hygrothermal = self.hygrothermal.scaled(f)
        new.element_hygrothermal = {k: v.scaled(f) for k, v in self.element_hygrothermal.items()}
        return new

    @classmethod
    def combine(cls, terms: Sequence[Tuple["LoadCase", float]], name: str = "combination") -> "LoadCase":
        """Linear combination sum(factor_i * case_i), hygrothermal states included."""
        out = cls(name)
        element_states: Dict[int, HygrothermalState] = {}
        ids = set()
        for case, _ in terms:
            ids |= set(case.element_hygrothermal)
        for case, factor in terms:
            s = case.scaled(factor)
            for k, v in s.nodal_loads.items():
                out.nodal_loads[k] = out.nodal_loads.get(k, 0.0) + v
            out.global_pressure = _sum_pressures([out.global_pressure, s.global_pressure])
            for e, p in s.element_pressures.items():
                out.element_pressures[e] = _sum_pressures([out.element_pressures.get(e, 0.0), p])
            out.line_loads += s.line_loads
            out.body_loads += s.body_loads
            out.hygrothermal = out.hygrothermal + s.hygrothermal
            for e in ids:
                element_states[e] = element_states.get(e, NULL_STATE) + s.state_for(e)
        for e in ids:
            out.element_hygrothermal[e] = element_states[e]
        return out


@dataclass(frozen=True)
class LoadCombination:
    """Named linear combination of load cases, e.g. 1.35 G + 1.5 Q."""

    name: str
    factors: Mapping[str, float]

    def terms(self, cases: Mapping[str, LoadCase]) -> List[Tuple[LoadCase, float]]:
        missing = [n for n in self.factors if n not in cases]
        if missing:
            raise KeyError(f"Combination '{self.name}' refers to unknown load cases {missing}")
        return [(cases[n], float(f)) for n, f in self.factors.items()]
