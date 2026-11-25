"""Per-ply failure field: the failure index of every ply of every shell element, and the governing ply.

``ply_failure_field`` evaluates a failure criterion from laminate-constitutive-model
(Tsai-Wu, Hashin, maximum stress, ...) on the ply stresses of a static solution. At each
recovery point of each element it samples every ply at the given positions through its
thickness (bottom, mid and top by default) and keeps, per element and ply, the largest
index, its strength ratio (the load factor to failure for a linear model), its failure
mode and where it occurs. Per element it also keeps the governing ply, the ply with the
largest index. The stresses come from the same recoverer as ``ResultsExtractor``, with
the generalised strains of each point computed once for all its plies.

``failure_envelope`` takes the field of several load cases and keeps, per element and
ply, the worst one and which case it came from.

Plies are numbered from the bottom of each element's laminate (``ply_bounds`` order). A
field over elements with different ply counts pads the missing plies with NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from laminate_constitutive_model.laminate_failure_criteria import FailureCriterion
from shell_fem.pre_processing.load_library.load_case import LoadCase
from shell_fem.processing.static.results.containers import SolutionResult
from shell_fem.processing.static.results.results_extractor import ResultsExtractor
from shell_fem.processing.static.results.stress_recovery import ExactStressRecoverer

DEFAULT_POSITIONS = ("bottom", "mid", "top")


@dataclass
class PlyFailureField:
    """Failure index per element and ply, padded with NaN to the largest ply count.

    ``index``, ``strength_ratio`` (n, P); ``mode`` (n, P) mode names ('' for padding);
    ``location`` (n, P, 3) the point of the largest index; ``z`` (n, P) its height
    through the laminate. ``n_plies`` (n,) is each element's own ply count.
    """

    criterion: str
    element_ids: np.ndarray
    centroid: np.ndarray
    n_plies: np.ndarray
    index: np.ndarray
    strength_ratio: np.ndarray
    mode: np.ndarray
    location: np.ndarray
    z: np.ndarray

    def __post_init__(self):
        self._row = {int(e): i for i, e in enumerate(self.element_ids)}

    @property
    def governing_ply(self) -> np.ndarray:
        """(n,) the ply with the largest index in each element (-1 for an element with no plies)."""
        filled = np.where(np.isnan(self.index), -np.inf, self.index)
        out = np.argmax(filled, axis=1)
        return np.where(self.n_plies > 0, out, -1)

    @property
    def max_index(self) -> np.ndarray:
        """(n,) the largest index over each element's plies."""
        return np.nanmax(self.index, axis=1)

    @property
    def min_strength_ratio(self) -> np.ndarray:
        """(n,) the smallest strength ratio over each element's plies (the element's reserve factor)."""
        return np.nanmin(self.strength_ratio, axis=1)

    @property
    def governing_mode(self) -> np.ndarray:
        rows = np.arange(self.element_ids.size)
        return self.mode[rows, np.maximum(self.governing_ply, 0)]

    @property
    def failed(self) -> np.ndarray:
        """(n,) elements with any ply index at or above 1."""
        return self.max_index >= 1.0 - 1e-9

    def row(self, element_id: int) -> int:
        return self._row[int(element_id)]

    def for_elements(self, element_ids: Sequence[int]) -> np.ndarray:
        """Row indices of ``element_ids`` (to read any field for an element set)."""
        return np.array([self._row[int(e)] for e in element_ids], dtype=int)

    def worst(self, element_ids: Optional[Sequence[int]] = None) -> Dict[str, object]:
        """The governing element, ply, index, strength ratio, mode and location (over a set, or all)."""
        rows = np.arange(self.element_ids.size) if element_ids is None else self.for_elements(element_ids)
        if rows.size == 0:
            raise ValueError("no elements to search")
        r = rows[int(np.nanargmax(self.max_index[rows]))]
        k = int(self.governing_ply[r])
        return {"element_id": int(self.element_ids[r]), "ply": k, "index": float(self.index[r, k]),
                "strength_ratio": float(self.strength_ratio[r, k]), "mode": str(self.mode[r, k]),
                "location": self.location[r, k].copy(), "z": float(self.z[r, k])}


def ply_failure_field(solution: SolutionResult, criterion: FailureCriterion, load_case: Optional[LoadCase] = None,
                      positions: Sequence = DEFAULT_POSITIONS, recoverer: Optional[ExactStressRecoverer] = None,
                      element_ids: Optional[Sequence[int]] = None) -> PlyFailureField:
    """Failure index of every ply of the shell elements of a static solution (see the module docstring).

    ``positions`` are places through each ply: "bottom", "mid", "top" or fractions in
    [0, 1]. ``element_ids`` limits the field to some elements. Every ply's material needs
    strength properties.
    """
    mesh = solution.mesh
    load_case = load_case if load_case is not None else (solution.load_case or LoadCase())
    recoverer = recoverer or ExactStressRecoverer(mesh.hygrothermal_engine)
    if not hasattr(recoverer, "recover_at_heights"):
        raise TypeError("the recoverer needs recover_at_heights (ExactStressRecoverer)")
    elements = list(mesh.shell_elements)
    if element_ids is not None:
        wanted = {int(e) for e in element_ids}
        elements = [e for e in elements if e.element_id in wanted]
    n = len(elements)
    P = max((e.laminate.n_plies for e in elements), default=0)
    index = np.full((n, P), np.nan)
    ratio = np.full((n, P), np.nan)
    mode = np.full((n, P), "", dtype=object)
    location = np.full((n, P, 3), np.nan)
    zs = np.full((n, P), np.nan)
    centroid = np.zeros((n, 3))
    n_plies = np.zeros(n, dtype=int)
    for j, element in enumerate(elements):
        laminate = element.laminate
        n_plies[j] = laminate.n_plies
        centroid[j] = element.centroid
        heights = [(k, *ResultsExtractor.ply_height(element, k, p)) for k in range(laminate.n_plies) for p in positions]
        strengths = []
        for ply in laminate.plies:
            strength = getattr(ply.material, "strength", None)
            if strength is None:
                raise ValueError(f"element {element.element_id}: ply material {ply.material!r} has no strengths")
            strengths.append(strength)
        u = solution.element_displacements(element.element_id)
        state = load_case.state_for(element.element_id)
        for xi, eta, _ in element.gauss_points():
            xyz = element.global_coordinates(xi, eta)
            states = recoverer.recover_at_heights(element, u, [(z, side) for _, z, side in heights], xi, eta, state)
            for (k, z, _), p in zip(heights, states):
                result = criterion.evaluate(p.material_stress_3d(), strengths[k])
                if not result.index <= index[j, k]:      # also replaces NaN
                    index[j, k] = result.index
                    ratio[j, k] = result.strength_ratio
                    mode[j, k] = result.mode.value
                    location[j, k] = xyz
                    zs[j, k] = z
    return PlyFailureField(type(criterion).__name__, np.array([e.element_id for e in elements], dtype=int), centroid,
                           n_plies, index, ratio, mode, location, zs)


@dataclass
class FailureEnvelope:
    """The worst of several load cases per element and ply: ``field`` holds the values, ``case`` (n, P) the names."""

    field: PlyFailureField
    case: np.ndarray

    @property
    def governing_case(self) -> np.ndarray:
        """(n,) the load case of each element's governing ply."""
        rows = np.arange(self.case.shape[0])
        return self.case[rows, np.maximum(self.field.governing_ply, 0)]


def failure_envelope(fields: Mapping[str, PlyFailureField]) -> FailureEnvelope:
    """Envelope of the ply failure fields of several cases on the same elements (the largest index wins)."""
    if not fields:
        raise ValueError("no failure fields to envelope")
    names = list(fields)
    first = fields[names[0]]
    for name in names[1:]:
        f = fields[name]
        if f.index.shape != first.index.shape or not np.array_equal(f.element_ids, first.element_ids):
            raise ValueError(f"field {name!r} is not on the same elements and plies")
    stacked = np.stack([np.where(np.isnan(fields[k].index), -np.inf, fields[k].index) for k in names])
    pick = np.argmax(stacked, axis=0)
    rows, cols = np.indices(pick.shape)

    def take(attr):
        values = np.stack([getattr(fields[k], attr) for k in names])
        return values[pick, rows, cols]

    location = np.stack([fields[k].location for k in names])[pick, rows, cols]
    field = PlyFailureField(first.criterion, first.element_ids.copy(), first.centroid.copy(), first.n_plies.copy(),
                            take("index"), take("strength_ratio"), take("mode"), location, take("z"))
    case = np.array(names, dtype=object)[pick]
    case[np.isnan(field.index)] = ""
    return FailureEnvelope(field, case)


__all__ = ["DEFAULT_POSITIONS", "FailureEnvelope", "PlyFailureField", "failure_envelope", "ply_failure_field"]
