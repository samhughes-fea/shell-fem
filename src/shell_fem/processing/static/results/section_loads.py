"""Section loads along a span: the six internal resultants at each cut, integrated from the shell.

A cut at span position z_k divides the model into an inboard part and an outboard part.
The outboard part is every element whose centroid is at or beyond z_k (an element lying
in the cut plane, such as a rib, belongs to it); the cut nodes are the nodes at z_k.

The internal load at the cut comes from the element stresses, not from the applied loads:
at each cut node, the outboard elements' internal forces K_e u_e less their own
distributed loads f_e are what the inboard part supplies,

    S_k = Σ_{cut nodes} Σ_{e outboard} (K_e u_e − f_e),

reduced to [Fx, Fy, Fz, Mx, My, Mz] about a point on the cut. ``internal`` is that
resultant: the load the inboard part applies to the outboard part. At the root cut it is
the root reaction.

``applied`` is the resultant, about the same point, of every load applied to the
outboard part: the assembled load vector at nodes beyond the cut plus the outboard
elements' distributed loads at the cut nodes. It is what a beam code gets by
integrating the loads per unit span from the tip in, V(z) = ∫_z^L f dζ and
M(z) = ∫_z^L (ζ − z) × f dζ, but taken from the loads the shell model actually carries.

Each cut is a free body, so ``internal + applied`` is zero to round-off in a converged
solve (``residual``). A residual well above round-off means the load path is broken,
for example a disconnected part or a load the cut cannot see. Nodal and line loads
that sit exactly on cut nodes count as inboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from shell_fem.pre_processing.load_library.load_case import LoadCase
from shell_fem.processing.static.operations.assembly import GlobalAssembler
from shell_fem.processing.static.results.containers import SolutionResult


def resultant_about(mesh, vector: np.ndarray, point: Sequence[float]) -> np.ndarray:
    """[Fx, Fy, Fz, Mx, My, Mz] of a nodal force vector (global DOFs) about ``point``."""
    nodal = mesh.standard_array(np.asarray(vector, dtype=float))
    r = mesh.coordinates_array() - np.asarray(point, dtype=float)
    force = nodal[:, :3].sum(axis=0)
    moment = np.cross(r, nodal[:, :3]).sum(axis=0) + nodal[:, 3:6].sum(axis=0)
    return np.concatenate([force, moment])


@dataclass
class SectionLoads:
    """Internal and applied resultants at each cut of one load case.

    ``stations`` (n,) are the cut positions on the span axis and ``points`` (n, 3) the
    points the moments are taken about. ``internal`` and ``applied`` are (n, 6),
    [Fx, Fy, Fz, Mx, My, Mz] in global axes.
    """

    stations: np.ndarray
    points: np.ndarray
    internal: np.ndarray
    applied: np.ndarray

    @property
    def residual(self) -> np.ndarray:
        """internal + applied at each cut: zero to round-off for a sound model."""
        return self.internal + self.applied

    @property
    def relative_residual(self) -> float:
        """Largest residual component over the largest internal component, across all cuts."""
        scale = float(np.abs(self.internal).max())
        return float(np.abs(self.residual).max()) / scale if scale > 0.0 else 0.0


class SectionLoadIntegrator:
    """Section cuts of one mesh at fixed span positions, reused for many load cases.

    ``stations`` must each pass through mesh nodes (a blade's ``span_positions`` do).
    Moments are taken about ``(reference[0], reference[1], z_k)`` at each cut, the
    span axis by default. Only the elements that touch a cut are evaluated.
    """

    def __init__(self, mesh, stations: Sequence[float], reference: Sequence[float] = (0.0, 0.0), axis: int = 2,
                 assembler: Optional[GlobalAssembler] = None, tol: float = 1e-9):
        if axis != 2:
            raise ValueError("section cuts are along z (axis=2)")
        self.mesh = mesh
        self.stations = np.asarray(stations, dtype=float)
        if self.stations.ndim != 1 or self.stations.size == 0:
            raise ValueError("stations must be a non-empty sequence of span positions")
        self.assembler = assembler or GlobalAssembler(sparse=False)
        xyz = mesh.coordinates_array()
        z = xyz[:, 2]
        self._tol = tol * max(1.0, float(np.abs(z).max()))
        ids = mesh.node_ids
        self._row = {n: i for i, n in enumerate(ids)}
        self._node_z = z
        self.points = np.column_stack([np.full(self.stations.size, float(reference[0])),
                                       np.full(self.stations.size, float(reference[1])), self.stations])
        self._cut_nodes, self._cut_elements = [], []
        centroids = {e.element_id: float(np.mean([z[self._row[n]] for n in e.node_ids])) for e in mesh.elements}
        for zk in self.stations:
            on_cut = np.abs(z - zk) <= self._tol
            if not on_cut.any():
                raise ValueError(f"no nodes at the cut z = {zk:g}; cut at span positions that pass through nodes")
            cut_ids = {ids[i] for i in np.flatnonzero(on_cut)}
            self._cut_nodes.append(on_cut)
            self._cut_elements.append([e for e in mesh.elements
                                       if centroids[e.element_id] >= zk - self._tol
                                       and any(n in cut_ids for n in e.node_ids)])
        touching = {e.element_id: e for elements in self._cut_elements for e in elements}
        self._stiffness = {eid: np.asarray(e.stiffness_matrix(), dtype=float) for eid, e in touching.items()}
        self._dofs = {eid: self.mesh.element_dofs(e) for eid, e in touching.items()}
        self._covered = mesh.covered_solids() if getattr(mesh, "has_solids", False) else set()
        self._node_dofs = [self._dofs_of(mask) for mask in self._cut_nodes]
        self._beyond = [self._dofs_of(z > zk + self._tol) for zk in self.stations]

    def _dofs_of(self, node_mask: np.ndarray) -> np.ndarray:
        ids = self.mesh.node_ids
        rows = [self.mesh.dof_index(ids[i], 0) + k for i in np.flatnonzero(node_mask)
                for k in range(self.mesh.node_dof_count(ids[i]))]
        return np.asarray(rows, dtype=int)

    def integrate(self, solution: SolutionResult, load_case: Optional[LoadCase] = None) -> SectionLoads:
        """Section loads of one solution; ``load_case`` defaults to ``solution.load_case``."""
        load_case = load_case if load_case is not None else solution.load_case
        if load_case is None:
            raise ValueError("section loads need the load case of the solution")
        n = self.mesh.n_dof
        u = np.asarray(solution.displacements, dtype=float)
        F = np.asarray(solution.load_vector, dtype=float)
        loads = {}
        internal = np.zeros((self.stations.size, 6))
        applied = np.zeros((self.stations.size, 6))
        for k, elements in enumerate(self._cut_elements):
            stress_side = np.zeros(n)
            element_loads = np.zeros(n)
            for e in elements:
                eid = e.element_id
                dofs = self._dofs[eid]
                if eid not in loads:
                    loads[eid] = self.assembler.element_load_vector(e, load_case, eid not in self._covered)
                stress_side[dofs] += self._stiffness[eid] @ u[dofs] - loads[eid]
                element_loads[dofs] += loads[eid]
            on_cut = np.zeros(n)
            on_cut[self._node_dofs[k]] = stress_side[self._node_dofs[k]]
            outboard = np.zeros(n)
            outboard[self._beyond[k]] = F[self._beyond[k]]
            outboard[self._node_dofs[k]] += element_loads[self._node_dofs[k]]
            internal[k] = resultant_about(self.mesh, on_cut, self.points[k])
            applied[k] = resultant_about(self.mesh, outboard, self.points[k])
        return SectionLoads(self.stations.copy(), self.points.copy(), internal, applied)


def section_loads(solution: SolutionResult, stations: Sequence[float], reference: Sequence[float] = (0.0, 0.0),
                  load_case: Optional[LoadCase] = None) -> SectionLoads:
    """Section loads of one static solution at the given span positions (see the module docstring)."""
    return SectionLoadIntegrator(solution.mesh, stations, reference).integrate(solution, load_case)


__all__ = ["SectionLoadIntegrator", "SectionLoads", "resultant_about", "section_loads"]
