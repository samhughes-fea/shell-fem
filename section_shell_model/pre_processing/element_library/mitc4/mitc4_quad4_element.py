"""MITC4 quad4 element returning formulation cache objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.section_shell_model.pre_processing.containers.formulation_objects import (
    ShellForceObject,
    ShellStiffnessObject,
)
from ..shell_element_base import ShellElementBase
from .element import mitc4_stiffness

_ELEMENT_TYPE = "MITC4-FSDT"
_U_X = 0
_U_S = 1
_W = 2


@dataclass(frozen=True)
class Mitc4ElementGeometry:
    L_s: float
    L_s_donnell: float
    L_x: float
    local_nodes: tuple[int, int, int, int]


@dataclass(frozen=True)
class Mitc4ElementMaterial:
    ABD: NDArray[np.float64]
    thickness: float
    G_eff: float


@dataclass(frozen=True)
class Mitc4ElementLoads:
    nx_avg: float
    nxy_avg: float
    kappa_avg: float | None = None


class Mitc4Quad4Element(ShellElementBase):
    """Production FSDT strip element (membrane Nx/Nxy + optional Donnell q_n)."""

    def __init__(
        self,
        *,
        element_id: int,
        panel_index: int,
        local_element_index: int,
        geometry: Mitc4ElementGeometry,
        material: Mitc4ElementMaterial,
        loads: Mitc4ElementLoads,
        dof_per_node: int = 6,
    ) -> None:
        self.element_id = element_id
        self.panel_index = panel_index
        self.local_element_index = local_element_index
        self.geometry = geometry
        self.material = material
        self.loads = loads
        self.dof_per_node = dof_per_node

    def element_stiffness_matrix(self) -> ShellStiffnessObject:
        g = self.geometry
        m = self.material
        ke = mitc4_stiffness(
            g.L_s,
            g.L_x,
            m.ABD,
            m.thickness,
            G_eff=m.G_eff,
            include_drilling=self.dof_per_node >= 6,
        )
        return ShellStiffnessObject(
            element_id=self.element_id,
            element_type=_ELEMENT_TYPE,
            K_e=np.asarray(ke, dtype=np.float64),
            panel_index=self.panel_index,
            local_element_index=self.local_element_index,
        )

    def element_force_vector(self) -> ShellForceObject:
        g = self.geometry
        ld = self.loads
        n = self.dof_per_node
        fe = np.zeros(4 * n, dtype=np.float64)

        f_node_x = ld.nx_avg * g.L_s / 2.0
        f_node_s = ld.nxy_avg * g.L_s / 2.0
        for node_slot in (2, 3):
            base = node_slot * n
            fe[base + _U_X] += f_node_x
            fe[base + _U_S] += f_node_s

        if ld.kappa_avg is not None:
            q_n = ld.nx_avg * ld.kappa_avg
            f_node_w = q_n * g.L_s_donnell * g.L_x / 4.0
            for node_slot in range(4):
                fe[node_slot * n + _W] += f_node_w

        return ShellForceObject(
            element_id=self.element_id,
            element_type=_ELEMENT_TYPE,
            F_e=fe,
            panel_index=self.panel_index,
            local_element_index=self.local_element_index,
        )


__all__ = [
    "Mitc4ElementGeometry",
    "Mitc4ElementLoads",
    "Mitc4ElementMaterial",
    "Mitc4Quad4Element",
]
