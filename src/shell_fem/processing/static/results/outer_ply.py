"""Outer-ply stress and strain from classical laminate theory.

At a height z from the element reference surface the CLPT in-plane strain is

    eps(z) = eps0 + z kappa

and the ply stress is Qbar (eps - eps_ht). z follows the laminate, including an
offset of the mid-plane from the reference surface, so an eccentric stack
(PCOMP Z0) is recovered on the mesh as meshed. The two outer faces are the
bottom of the first ply and the top of the last ply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from laminate_constitutive_model.laminate_hygrothermal import NULL_STATE
from shell_fem.pre_processing.load_library.load_case import LoadCase
from shell_fem.processing.static.results.containers import SolutionResult


@dataclass(frozen=True)
class OuterPlyField:
    """CLPT stress and strain at the outer face of the bottom and top plies.

    One row per shell element per face (``"bottom"`` then ``"top"``). Strains and
    stresses are the in-plane components [xx, yy, xy]. ``strain_global`` is the
    total CLPT strain eps0 + z kappa; ``strain_mechanical`` is that strain with
    the free hygrothermal expansion removed, and it is the strain that produces
    ``stress_global``. Material-axis arrays are in the ply fibre axes.
    ``z`` is the face height from the element reference surface. ``xyz`` is the
    point on that reference surface where the strain was read (the element centroid).
    """

    element_ids: np.ndarray
    face: np.ndarray
    ply_index: np.ndarray
    z: np.ndarray
    xyz: np.ndarray
    strain_global: np.ndarray
    strain_mechanical: np.ndarray
    strain_material: np.ndarray
    stress_global: np.ndarray
    stress_material: np.ndarray


def outer_ply_stress_strain(solution: SolutionResult, load_case: Optional[LoadCase] = None) -> OuterPlyField:
    """Outer-ply stress and strain of a static solution, from CLPT recovery.

    Evaluated at each shell element's centroid. The element may have been solved
    with any kinematic model; the in-plane recovery used here is the classical
    one, eps(z) = eps0 + z kappa, not an enriched through-thickness strain.
    """
    load_case = load_case if load_case is not None else solution.load_case
    rows = {k: [] for k in ("eid", "face", "ply", "z", "xyz", "eg", "em", "emat", "sg", "sm")}
    shells = solution.mesh.shell_elements
    if not shells:
        raise ValueError("the mesh has no shell elements")
    for element in shells:
        u = solution.element_displacements(element.element_id)
        xi, eta = element.NATURAL_CENTROID
        eps0, kappa, _ = element.generalized_strains(u, xi, eta)
        state = load_case.state_for(element.element_id) if load_case is not None else NULL_STATE
        engine = element.hygrothermal_engine
        xyz = element.centroid
        bounds = element.laminate.ply_bounds()
        faces = (
            ("bottom", 0, bounds[0][0]),
            ("top", element.laminate.n_plies - 1, bounds[-1][1]),
        )
        for face, ply_index, z in faces:
            ply = element.laminate.plies[ply_index]
            eps = eps0 + z * kappa
            eps_ht = np.zeros(3) if state.is_null else engine.global_free_strain(ply, state, z, element.laminate.offset)
            eps_m = eps - eps_ht
            sigma = ply.Q_bar @ eps_m
            for key, value in zip(
                rows,
                (element.element_id, face, ply_index, z, xyz, eps, eps_m,
                 ply.to_material_strain(eps_m), sigma, ply.to_material_stress(sigma)),
            ):
                rows[key].append(value)
    a = {k: np.array(v) for k, v in rows.items()}
    return OuterPlyField(
        a["eid"], a["face"], a["ply"], a["z"].astype(float), a["xyz"],
        a["eg"], a["em"], a["emat"], a["sg"], a["sm"],
    )
