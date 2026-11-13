"""Boundary conditions: prescribed DOF values, multi-point constraints and grounded springs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from shell_fem.pre_processing.containers.mesh import ALL_DOFS, DofKey, DofLike, EnrichedDOF


@dataclass(frozen=True)
class LinearConstraint:
    """Multi-point constraint u_slave = sum(coef_i * u_master_i) + constant."""

    slave: DofKey
    masters: Tuple[Tuple[DofKey, float], ...]
    constant: float = 0.0


def _dof_key(node: int, dof: DofLike) -> DofKey:
    return (int(node), dof if isinstance(dof, EnrichedDOF) else int(dof))


class BoundaryConditions:
    """Prescribed DOF values, linear multi-point constraints and grounded springs."""

    def __init__(self):
        self.prescribed: Dict[DofKey, float] = {}
        self.constraints: List[LinearConstraint] = []
        self.springs: Dict[Tuple[int, int], float] = {}

    def fix(self, node_ids: Iterable[int], dofs: Iterable[DofLike] = ALL_DOFS, value: float = 0.0) -> "BoundaryConditions":
        dofs = tuple(dofs)
        for n in node_ids:
            for d in dofs:
                self.prescribed[_dof_key(n, d)] = float(value)
        return self

    def clamp(self, node_ids: Iterable[int]) -> "BoundaryConditions":
        """Fix all six shell DOFs and every enrichment amplitude."""
        return self.fix(node_ids, ALL_DOFS + (EnrichedDOF.ALL,))

    def add_constraint(
        self, slave: DofKey, masters: Sequence[Tuple[DofKey, float]], constant: float = 0.0
    ) -> "BoundaryConditions":
        slave = (int(slave[0]), int(slave[1]))
        masters = tuple(((int(k[0]), int(k[1])), float(c)) for k, c in masters)
        self.constraints.append(LinearConstraint(slave, masters, float(constant)))
        return self

    def tie(self, slave: DofKey, master: DofKey) -> "BoundaryConditions":
        return self.add_constraint(slave, [(master, 1.0)])

    def fix_direction(
        self, node_ids: Iterable[int], direction: Sequence[float], rotation: bool = False, value: float = 0.0
    ) -> "BoundaryConditions":
        """Skew support: the displacement (or rotation) component along ``direction`` equals ``value``.

        Written as the constraint n . u = value with the component of largest |n| as slave,
        so supports on inclined or curved edges need no local coordinate systems.
        """
        n = np.asarray(direction, dtype=float)
        n = n / np.linalg.norm(n)
        base = 3 if rotation else 0
        s = int(np.argmax(np.abs(n)))
        for node in node_ids:
            masters = [((int(node), base + k), -n[k] / n[s]) for k in range(3) if k != s and n[k] != 0.0]
            self.add_constraint((int(node), base + s), masters, value / n[s])
        return self

    def add_spring(self, node_ids: Iterable[int], dofs: Iterable[int], stiffness: float) -> "BoundaryConditions":
        """Grounded spring of the given stiffness on each listed DOF (adds to existing springs)."""
        if stiffness < 0.0:
            raise ValueError("spring stiffness must be non-negative")
        dofs = tuple(int(d) for d in dofs)
        for node in node_ids:
            for d in dofs:
                key = (int(node), d)
                self.springs[key] = self.springs.get(key, 0.0) + float(stiffness)
        return self

    def constrained_keys(self) -> set:
        return set(self.prescribed) | {c.slave for c in self.constraints}

    def numeric_prescribed(self) -> Dict[Tuple[int, int], float]:
        """Prescribed values on the six shell DOFs only (what external solver decks can express)."""
        return {k: v for k, v in self.prescribed.items() if not isinstance(k[1], EnrichedDOF)}
