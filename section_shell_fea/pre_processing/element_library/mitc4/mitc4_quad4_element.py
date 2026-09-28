"""MITC4 quad4 element returning formulation cache objects."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

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


def _f64_bits(value: float) -> bytes:
    """Exact little-endian float64 bytes (no rounding) for template identity."""
    return struct.pack("<d", float(value))


def _stiffness_template_key(
    *, L_s: float, L_x: float, ndof: int, material: "Mitc4ElementMaterial"
) -> tuple:
    """Exact per-build reuse key for one element's ``K_e``.

    Bitwise-exact (float64 bytes, never ``round()``) so within-build reuse is
    identical to a fresh assembly (WP0 exactness). Covers every quantity that
    ``K_e`` depends on: element lengths, DOF count, and the constitutive blocks
    (typed ``fsdt`` A/B/D/A_s, or the legacy ABD tuple + thickness + G_eff).
    """
    parts: list[object] = [_f64_bits(L_s), _f64_bits(L_x), int(ndof)]
    if material.fsdt is not None:
        f = material.fsdt
        parts += [
            b"fsdt",
            np.ascontiguousarray(f.A, dtype=np.float64).tobytes(),
            np.ascontiguousarray(f.B, dtype=np.float64).tobytes(),
            np.ascontiguousarray(f.D, dtype=np.float64).tobytes(),
            np.ascontiguousarray(f.A_s, dtype=np.float64).tobytes(),
            _f64_bits(material.thickness),
        ]
    else:
        parts += [
            b"abd",
            np.ascontiguousarray(material.ABD, dtype=np.float64).tobytes(),
            _f64_bits(material.thickness),
            _f64_bits(material.G_eff),
        ]
    return tuple(parts)


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
        template_cache: Any | None = None,
    ) -> None:
        self.element_id = element_id
        self.panel_index = panel_index
        self.local_element_index = local_element_index
        self.geometry = geometry
        self.material = material
        self.loads = loads
        self.dof_per_node = dof_per_node
        self.template_cache = template_cache

    def _assemble_ke(self) -> NDArray[np.float64]:
        g = self.geometry
        m = self.material
        shape = ShapeFunctionOperator(g.L_s, g.L_x)
        material = MaterialStiffnessOperator(
            m.fsdt if m.fsdt is not None else m.ABD,
            m.thickness,
            G_eff=None if m.fsdt is not None else m.G_eff,
        )
        return np.asarray(
            assemble_mitc4_stiffness(shape, material, ndof=self.dof_per_node).K_e,
            dtype=np.float64,
        )

    def element_stiffness_matrix(self) -> ShellStiffnessObject:
        g = self.geometry
        cache = self.template_cache
        if cache is not None:
            # Exact within-build reuse: identical geometry+constitutive elements
            # (uniform meshes, shared panel material) build ``K_e`` once. The key
            # is computed without constructing the operators, so a hit reuses the
            # ShapeFunctionOperator / MaterialStiffnessOperator work as well.
            key = _stiffness_template_key(
                L_s=g.L_s, L_x=g.L_x, ndof=self.dof_per_node, material=self.material
            )
            ke = cache.get_or_build(key, self._assemble_ke)
        else:
            ke = self._assemble_ke()
        return ShellStiffnessObject(
            element_id=self.element_id,
            element_type=_ELEMENT_TYPE,
            K_e=np.asarray(ke, dtype=np.float64),
            panel_index=self.panel_index,
            local_element_index=self.local_element_index,
            L_s=float(g.L_s),
            L_s_donnell=float(g.L_s_donnell),
            L_x=float(g.L_x),
            formulation_id=_ELEMENT_TYPE,
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
            L_s=float(g.L_s),
            L_s_donnell=float(g.L_s_donnell),
            L_x=float(g.L_x),
            formulation_id=_ELEMENT_TYPE,
        )


__all__ = [
    "Mitc4ElementGeometry",
    "Mitc4ElementLoads",
    "Mitc4ElementMaterial",
    "Mitc4Quad4Element",
]
