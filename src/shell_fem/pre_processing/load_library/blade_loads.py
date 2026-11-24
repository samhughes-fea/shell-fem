"""Loads and supports for a full blade model, given as distributions along the span.

The span runs along global z, as in ``BladeMesh``. Loads from an aeroelastic or
hydrodynamic code usually come as sectional quantities per unit span. These
functions turn them into surface loads on the blade's shell elements:

``add_sectional_force``  force per unit span [fx, fy, fz](z) (N/m, global axes) acting at
                         a point of the section (the pitch axis x = y = 0 by default),
                         spread over the skin of each span bay as a uniform traction plus
                         the self-balanced traction that moves it from the skin centroid
                         to that point
``add_sectional_torque`` twisting moment per unit span m(z) (N m/m about z), applied as
                         a tangential shear flow about the centroid of the skin in each
                         bay, so it carries no net force
``add_spanwise_pressure`` pressure p(z) (Pa) along the element normals of a set
``clamp_root``            every DOF of the root nodes fixed

Gravity is ``LoadCase.add_body_load`` and rotation ``LoadCase.add_centrifugal``.

A bay is the part of the mesh between consecutive ``BladeMesh.span_positions`` (or
between distinct node z values when the blade has none). A uniform traction round the
section acts at the skin centroid c of the bay, which on an aerofoil lies well aft of the
pitch axis. ``add_sectional_force`` therefore adds, in each bay, a traction with no net
force that carries the moment (a - c) x f of the offset to the application point a: a
shear flow for the torque about z and an axial traction linear over the section for the
bending moments of an axial force. The resultant of the bay then matches a beam model
with the force at a. The local distribution round the section is still a smeared one;
use a pressure field where the chordwise distribution itself matters.

A distribution is a function of z, or a table ``spanwise(z, values)`` interpolated
linearly and held at its end values beyond the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
from shell_fem.pre_processing.load_library.load_case import LoadCase

Distribution = Union[Callable[[float], object], "SpanwiseTable"]


@dataclass(frozen=True)
class SpanwiseTable:
    """Values (scalars or vectors) tabulated at span positions ``z``, interpolated linearly."""

    z: np.ndarray
    values: np.ndarray

    def __call__(self, z: float):
        v = self.values
        if v.ndim == 1:
            return float(np.interp(z, self.z, v))
        return np.array([np.interp(z, self.z, v[:, k]) for k in range(v.shape[1])])


def spanwise(z: Sequence[float], values) -> SpanwiseTable:
    """A distribution tabulated at increasing span positions ``z`` (values: (n,) or (n, k))."""
    z = np.asarray(z, dtype=float).ravel()
    values = np.asarray(values, dtype=float)
    if z.size < 2 or np.any(np.diff(z) <= 0.0):
        raise ValueError("z must hold at least two increasing span positions")
    if values.shape[0] != z.size:
        raise ValueError("values need one row per span position")
    return SpanwiseTable(z, values)


@dataclass(frozen=True)
class SpanBay:
    """The skin of one span bay: its elements, area, the centroid of the skin and its polar moment."""

    z0: float
    z1: float
    element_ids: tuple
    area: float
    centroid: np.ndarray  # (x, y) of the skin area
    polar_moment: float   # integral of the squared distance from the centroid, over the skin area
    second_moment: np.ndarray = None  # (2, 2) integral of r r^T over the skin area, r = (x, y) - centroid

    @property
    def length(self) -> float:
        return self.z1 - self.z0


def _element_ids(blade, element_set: Union[str, Sequence[int]]) -> List[int]:
    if isinstance(element_set, str):
        return list(blade.elements(element_set))
    return [int(e) for e in element_set]


def span_bays(blade, element_set: Union[str, Sequence[int]] = "skin") -> List[SpanBay]:
    """The span bays of ``blade`` with the elements of ``element_set`` in each (by element centroid)."""
    mesh = blade.mesh
    edges = np.asarray(getattr(blade, "span_positions", ()), dtype=float)
    if edges.size < 2:
        edges = np.unique(np.round(mesh.coordinates_array()[:, 2], 12))
    if edges.size < 2:
        raise ValueError("the blade has no extent along z")
    sums: Dict[int, np.ndarray] = {}
    members: Dict[int, List[int]] = {}
    for eid in _element_ids(blade, element_set):
        element = mesh.element(eid)
        acc = np.zeros(7)  # area, int x, int y, int z, int x^2, int y^2, int xy
        for xi, eta, w in element.load_rule():
            dA = element.jacobian(xi, eta)[1] * w
            x = element.global_coordinates(xi, eta)
            acc += dA * np.array([1.0, x[0], x[1], x[2], x[0] ** 2, x[1] ** 2, x[0] * x[1]])
        k = int(np.clip(np.searchsorted(edges, acc[3] / acc[0]) - 1, 0, edges.size - 2))
        sums[k] = sums.get(k, np.zeros(7)) + acc
        members.setdefault(k, []).append(eid)
    bays = []
    for k in sorted(sums):
        A, sx, sy, _, sxx, syy, sxy = sums[k]
        c = np.array([sx, sy]) / A
        S = np.array([[sxx, sxy], [sxy, syy]]) - A * np.outer(c, c)
        bays.append(SpanBay(float(edges[k]), float(edges[k + 1]), tuple(members[k]), float(A), c,
                            float(np.trace(S)), S))
    return bays


def _as_function(distribution: Distribution) -> Callable[[float], object]:
    if not callable(distribution):
        raise TypeError("a distribution is a function of z or spanwise(z, values)")
    return distribution


def add_sectional_force(load_case: LoadCase, blade, force_per_length: Distribution,
                        element_set: Union[str, Sequence[int]] = "skin",
                        point: Union[None, Sequence[float], Callable[[float], Sequence[float]]] = (0.0, 0.0)
                        ) -> LoadCase:
    """A force per unit span [fx, fy, fz](z) acting at ``point`` of each section, spread over the skin.

    ``point`` is the application point (x, y), fixed or a function of z: the pitch axis
    (0, 0) by default. In each bay the traction is

        t = f / P + (m_z / J) e_z x r + (g . r) e_z,     r = (x, y) - c,

    with P the skin area of the bay over its length (the mean perimeter), c the skin
    centroid, J and S its polar and second moments of area over the length, and
    m = (a - c) x f the moment that moves the force from c to the application point a.
    The shear-flow term carries m_z and the axial term, with S g = (-m_y, m_x), carries
    the bending moments of an axial force; neither has a net force. So the bay's force
    is the integral of f over it, and its moment is that of f at a, wherever the section
    is uniform along the bay.

    ``point=None`` leaves the force at the skin centroid (the uniform traction alone).
    """
    f = _as_function(force_per_length)
    if point is None:
        at = None
    elif callable(point):
        at = point
    else:
        fixed = np.asarray(point, dtype=float).reshape(2)
        at = lambda z, a=fixed: a  # noqa: E731
    for bay in span_bays(blade, element_set):
        P = bay.area / bay.length
        if at is None:
            load_case.add_traction(lambda xyz, f=f, P=P: np.asarray(f(xyz[2]), dtype=float) / P, bay.element_ids)
            continue
        J = bay.polar_moment / bay.length
        S_inv = np.linalg.pinv(bay.second_moment / bay.length)
        c = bay.centroid

        def traction(xyz, f=f, P=P, J=J, S_inv=S_inv, c=c):
            z = xyz[2]
            fz = np.asarray(f(z), dtype=float).reshape(3)
            d = np.asarray(at(z), dtype=float).reshape(2) - c       # a - c
            m = np.array([d[1] * fz[2], -d[0] * fz[2], d[0] * fz[1] - d[1] * fz[0]])  # (a - c, 0) x f
            g = S_inv @ np.array([-m[1], m[0]])
            r = np.array([xyz[0] - c[0], xyz[1] - c[1]])
            return fz / P + m[2] / J * np.array([-r[1], r[0], 0.0]) + np.array([0.0, 0.0, g @ r])

        load_case.add_traction(traction, bay.element_ids)
    return load_case


def add_sectional_torque(load_case: LoadCase, blade, torque_per_length: Distribution,
                         element_set: Union[str, Sequence[int]] = "skin") -> LoadCase:
    """Apply a twisting moment per unit span m(z) (about +z) as a shear flow round the skin.

    In each bay the traction is m(z) e_z x (x - c) / J, with c the centroid of the skin
    of the bay and J its polar moment over its length. Its moment about z is m per unit
    span and, for m uniform along the bay, it has no net force.
    """
    m = _as_function(torque_per_length)
    for bay in span_bays(blade, element_set):
        J = bay.polar_moment / bay.length
        c = bay.centroid

        def traction(xyz, m=m, J=J, c=c):
            rx, ry = xyz[0] - c[0], xyz[1] - c[1]
            return float(m(xyz[2])) / J * np.array([-ry, rx, 0.0])

        load_case.add_traction(traction, bay.element_ids)
    return load_case


def add_spanwise_pressure(load_case: LoadCase, blade, pressure: Distribution,
                          element_set: Union[str, Sequence[int]] = "skin") -> LoadCase:
    """Pressure p(z) along the element normals of ``element_set`` (positive along +e3)."""
    p = _as_function(pressure)
    return load_case.add_pressure(lambda xyz, p=p: float(p(xyz[2])), _element_ids(blade, element_set))


def clamp_root(blade, bcs: Optional[BoundaryConditions] = None) -> BoundaryConditions:
    """Fix every DOF of the root nodes (a rigid root)."""
    bcs = bcs if bcs is not None else BoundaryConditions()
    return bcs.clamp(blade.nodes("root"))


__all__ = ["SpanBay", "SpanwiseTable", "add_sectional_force", "add_sectional_torque", "add_spanwise_pressure",
           "clamp_root", "span_bays", "spanwise"]
