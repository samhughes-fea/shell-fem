"""Ply-resolution CLT recovery payload (tier 3 native output).

Adds a "ply" resolution alongside assembled / elemental / nodal / gaussian.
Frames (see ``geometry_utilities.coordinate_frames``):

* **E** (element = laminate): mid-surface ``eps0`` / ``kappa``, interface ``tau_iface``
* **M** (ply material): ``eps_ply_mat`` / ``sig_ply_mat`` after ``T_ply`` / ``T_ply_eps``

Failure criteria consume **M** only. Isotropic panels have M ≡ E (``n_ply=1``).
Validated ``FrameTag`` fields (ARCH-MUST-016 / FEA-R-001) make that contract
machine-checkable at the package boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blade_structure_utilities.geometry_utilities.coordinate_frames import (
    FRAME_E,
    FRAME_M,
    FrameTag,
    require_frame_tag,
)

# Face axis on ``eps_ply_mat`` / ``sig_ply_mat``: 0=outer (x3+), 1=mid, 2=inner (x3-).
# Blade-structure reading of outer/inner depends on panel kind (skin / spar / web);
# see compute_tertiary/docs/README.md § "Panel face semantics".
FACE_OUTER = 0
FACE_MID = 1
FACE_INNER = 2


@dataclass
class ShellPlyResults:
    """Per-element CLT recovery: E mid-surface kinematics + M face stresses."""

    eps0: np.ndarray | None = None
    """(n_elem, 3) mid-surface strains in **E** ``[ε11, ε22, γ12]`` from ABD."""

    kappa: np.ndarray | None = None
    """(n_elem, 3) curvatures in **E** ``[κ11, κ22, κ12]`` from ABD."""

    eps_ply_mat: np.ndarray | None = None
    """(n_elem, n_ply, 3, 3) per-plate face strains in **M**
    ``[ε11, ε22, γ12]`` at outer/mid/inner."""

    sig_ply_mat: np.ndarray | None = None
    """(n_elem, n_ply, 3, 3) per-plate face stresses in **M**
    ``[σ11, σ22, τ12]`` at outer/mid/inner.

    Face axis: ``0`` = outer (higher \\(x_3\\)), ``1`` = mid, ``2`` = inner (lower \\(x_3\\)).

    Blade-structure aliases (when shell ``+x3`` follows the panel convention):
    skin outer/inner = fluid/void; spar-cap outer/inner = skin/void;
    web free surfaces = fore (LE side) / aft (TE side), not fluid/void.
    """

    trusted_mask: np.ndarray | None = None
    """(n_elem,) bool — provenance-trusted rows that entered the CLT solve."""

    tau_iface: np.ndarray | None = None
    """(n_elem, n_iface, 2) interface ``[τ13, τ23]`` [Pa] in **E**; ``n_iface = n_ply + 1``.

    Free surfaces are zero; interior values come from FSDT parabolic recovery
    of laminate-frame \\(Q_1,Q_2\\). Untrusted interlaminar rows remain zero with
    ``shear_trusted_mask`` false.
    """

    shear_trusted_mask: np.ndarray | None = None
    """(n_elem,) bool — rows whose \\(Q_1,Q_2\\) (secondary ``Q1``/``Q2``) passed gating."""

    sig_ply_mat_canonical: np.ndarray | None = None
    """(n_elem, n_ply, 3, 6) dense M-frame stress
    ``[σ11, σ22, σ33, τ23, τ13, τ12]``. MITC4 stores ``σ33=0`` as a
    not-modelled weak-form zero; capability is not inferred from the value."""

    stress_capability: np.ndarray | None = None
    """(6,) object array of capability tags matching canonical stress order."""

    physical_ply_count: np.ndarray | None = None
    """(n_elem,) physical ply counts; padded columns are storage only."""

    formulation_id: np.ndarray | None = None
    """(n_elem,) discrete formulation id per row."""

    element_key: np.ndarray | None = None
    """(n_elem, 3) ``[element_id, panel_index, local_element_index]``."""

    mid_surface_frame: FrameTag = FRAME_E
    """Frame of ``eps0`` / ``kappa`` (and ``tau_iface``)."""

    ply_material_frame: FrameTag = FRAME_M
    """Frame of ``eps_ply_mat`` / ``sig_ply_mat`` / ``sig_ply_mat_canonical``."""

    def __post_init__(self) -> None:
        self.mid_surface_frame = require_frame_tag(self.mid_surface_frame)
        self.ply_material_frame = require_frame_tag(self.ply_material_frame)
        if self.mid_surface_frame != FRAME_E:
            raise ValueError(
                f"ShellPlyResults.mid_surface_frame must be {FRAME_E!r}; "
                f"got {self.mid_surface_frame!r}"
            )
        if self.ply_material_frame != FRAME_M:
            raise ValueError(
                f"ShellPlyResults.ply_material_frame must be {FRAME_M!r}; "
                f"got {self.ply_material_frame!r}"
            )


__all__ = ["FACE_INNER", "FACE_MID", "FACE_OUTER", "ShellPlyResults"]
