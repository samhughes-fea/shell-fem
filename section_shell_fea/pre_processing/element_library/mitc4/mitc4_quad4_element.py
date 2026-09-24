"""MITC4 quad4 element returning formulation cache objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.material_utilities.laminate import (
    FSDTLaminateStiffness,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    ShellForceObject,
    ShellStiffnessObject,
)
from ..shell_element_base import ShellElementBase
from .element import assemble_mitc4_stiffness
from .utilities import (
    ElementLoadOperator,
    MaterialStiffnessOperator,
    ShapeFunctionOperator,
)

_ELEMENT_TYPE = "MITC4-FSDT"


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
    fsdt: FSDTLaminateStiffness | None = None


@dataclass(frozen=True)
class Mitc4ElementLoads:
    nx_avg: float
    nxy_avg: float
    kappa_avg: float | None = None


class Mitc4Quad4Element(ShellElementBase):
    """Production FSDT strip element (membrane N11/N12 + optional Donnell q_n)."""

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
        shape = ShapeFunctionOperator(g.L_s, g.L_x)
        material = MaterialStiffnessOperator(
            m.fsdt if m.fsdt is not None else m.ABD,
            m.thickness,
            G_eff=None if m.fsdt is not None else m.G_eff,
        )
        ke = assemble_mitc4_stiffness(
            shape, material, ndof=self.dof_per_node
        ).K_e
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
        shape = ShapeFunctionOperator(g.L_s, g.L_x)
        donnell_ratio = g.L_s_donnell / max(g.L_s, 1.0e-30)
        fe = ElementLoadOperator(shape, ndof=n).legacy_strip_load(
            nx=ld.nx_avg,
            nxy=ld.nxy_avg,
            kappa=ld.kappa_avg,
            donnell_area_ratio=donnell_ratio,
        )

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
