"""
MITC4 / MITC4+ flat-shell element for panel-level stress recovery.

Reference: Bathe & Dvorkin (1985), "A four-node plate bending element based on
Mindlin/Reissner plate theory and a mixed interpolation of tensorial components".

Curvilinear strip frame C (coincides with element/laminate E on each facet)
---------------------------------------------------------------------------
See geometry_utilities.coordinate_frames § Curvilinear strip (C).

  Parent (isoparametric — not a structural frame):
    xi  in [-1, 1]  -- along contour tangent s (= E x2), length L_s
    eta in [-1, 1]  -- along span x (= E x1), length L_x = 1 for unit slice
    zeta            -- through-thickness n (= E x3); not discretised

Node ordering (counter-clockwise):
  1: (xi=-1, eta=-1)   2: (xi=+1, eta=-1)
  3: (xi=+1, eta=+1)   4: (xi=-1, eta=+1)

DOFs per node (curvilinear C):
  MITC4+ (ndof=6, default): [u_x, u_s, w, beta_s, beta_x, beta_z] -> 24 DOFs/element
  MITC4  (ndof=5):          [u_x, u_s, w, beta_s, beta_x]        -> 20 DOFs/element

  u_x    : spanwise displacement (C x / E x1)
  u_s    : contour-tangential displacement (C s / E x2)
  w      : shell-normal displacement (C n / E x3)
  beta_s : compatibility rotation; theta1=beta_s  (~=  dw/dx in thin limit)
  beta_x : compatibility rotation; theta2=-beta_x (~= -dw/ds in thin limit)
  beta_z : drilling rotation about shell normal (MITC4+ only, Hughes-Brezzi stabilisation)

DOF ordering inside the n-vector (n = 4*ndof):
  [node1_ux, node1_us, node1_w, node1_bs, node1_bx, (node1_bz,)
   node2_ux, ..., node4_bx, (node4_bz)]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Gauss quadrature rules
# ---------------------------------------------------------------------------

def _gauss(n: int) -> tuple[NDArray, NDArray]:
    """Points and weights for n-point 1-D Gauss-Legendre on [-1, 1]."""
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    if n == 2:
        p = 1.0 / np.sqrt(3.0)
        return np.array([-p, p]), np.array([1.0, 1.0])
    if n == 3:
        p = np.sqrt(3.0 / 5.0)
        return np.array([-p, 0.0, p]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    if n == 4:
        p1 = np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0))
        p2 = np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0))
        w1 = (18.0 + np.sqrt(30.0)) / 36.0
        w2 = (18.0 - np.sqrt(30.0)) / 36.0
        return np.array([-p2, -p1, p1, p2]), np.array([w2, w1, w1, w2])
    raise ValueError(f"Gauss order {n} not implemented (use 1, 2, 3, or 4)")


# ---------------------------------------------------------------------------
# B-matrix construction
# ---------------------------------------------------------------------------

def _b_membrane(xi: float, eta: float, L_s: float, L_x: float, ndof: int = 6) -> NDArray:
    """
    Membrane B matrix (3 x 4*ndof).

    Strains in **C**/**E**: ``[ε11, ε22, γ12]`` ≡ ``[eps_xx, eps_ss, gamma_xs]``
    from ``u_x``, ``u_s`` DOFs.
    Jacobian: d/dx = (2/L_x)*d/deta,  d/ds = (2/L_s)*d/dxi.
    beta_z (DOF 5, ndof=6) does not contribute to membrane strain.
    """
    from .utilities.B_matrix import StrainDisplacementOperator
    from .utilities.shape_functions import ShapeFunctionOperator

    operator = StrainDisplacementOperator(
        ShapeFunctionOperator(L_s, L_x), ndof=ndof
    )
    return operator.natural_coordinate_form(
        np.array([xi]), np.array([eta])
    )[0][0]


def _b_bending(xi: float, eta: float, L_s: float, L_x: float, ndof: int = 6) -> NDArray:
    """
    Bending B matrix (3 x 4*ndof).

    Curvatures: ``kappa=-L_m theta`` with
    ``theta1=beta_s`` and ``theta2=-beta_x``.
    beta_z (DOF 5, ndof=6) does not contribute to bending curvatures.
    """
    from .utilities.B_matrix import StrainDisplacementOperator
    from .utilities.shape_functions import ShapeFunctionOperator

    operator = StrainDisplacementOperator(
        ShapeFunctionOperator(L_s, L_x), ndof=ndof
    )
    return operator.natural_coordinate_form(
        np.array([xi]), np.array([eta])
    )[1][0]


def _b_shear_mitc(xi: float, eta: float, L_s: float, L_x: float, ndof: int = 6) -> NDArray:
    """
    MITC4 mixed-interpolated transverse shear B matrix (2 x 4*ndof).

    Shear strains: [gamma_xn, gamma_sn]  (MITC tying, Bathe & Dvorkin 1985).
    beta_z (DOF 5, ndof=6) does not contribute to transverse shear.
    """
    from .utilities.B_matrix import StrainDisplacementOperator
    from .utilities.shape_functions import ShapeFunctionOperator

    operator = StrainDisplacementOperator(
        ShapeFunctionOperator(L_s, L_x), ndof=ndof
    )
    return operator.mitc_shear(np.array([xi]), np.array([eta]))[0]


def _b_drilling(xi: float, eta: float, L_s: float, L_x: float) -> NDArray:
    """
    Drilling strain-displacement matrix (1 x 24) for MITC4+ (ndof=6).

    Drilling strain: omega_drill = du_s/dx - du_x/ds - beta_z
    Penalised by K_drill = alpha_drill * G_eff * t * A_elem * B_drill^T @ B_drill.

    B_drill row for node I:
      u_x  col (base+0): -dN_I/ds
      u_s  col (base+1): +dN_I/dx
      beta_z col (base+5): -N_I
    """
    from .utilities.B_matrix import StrainDisplacementOperator
    from .utilities.shape_functions import ShapeFunctionOperator

    operator = StrainDisplacementOperator(ShapeFunctionOperator(L_s, L_x), ndof=6)
    drilling = operator.natural_coordinate_form(
        np.array([xi]), np.array([eta])
    )[3]
    assert drilling is not None
    return drilling[0]


# ---------------------------------------------------------------------------
# Element stiffness
# ---------------------------------------------------------------------------

_KE_TEMPLATE_CACHE: dict[tuple, NDArray] = {}


def _abd_fingerprint(abd: NDArray) -> tuple:
    return tuple(float(x) for x in np.asarray(abd, dtype=np.float64).ravel().round(12))


@dataclass(frozen=True, slots=True)
class Mitc4StiffnessContributions:
    """Auditable material and stabilization contributions to ``K_e``."""

    K_membrane: NDArray[np.float64]
    K_membrane_bending: NDArray[np.float64]
    K_bending: NDArray[np.float64]
    K_shear: NDArray[np.float64]
    K_drilling: NDArray[np.float64]

    @property
    def K_e(self) -> NDArray[np.float64]:
        return (
            self.K_membrane
            + self.K_membrane_bending
            + self.K_bending
            + self.K_shear
            + self.K_drilling
        )


def assemble_mitc4_stiffness(
    shape: object,
    material: object,
    *,
    ndof: int = 6,
) -> Mitc4StiffnessContributions:
    """Assemble named contributions from one geometry and one ABDAs object."""
    from .utilities.B_matrix import StrainDisplacementOperator
    from .utilities.D_matrix import MaterialStiffnessOperator
    from .utilities.shape_functions import ShapeFunctionOperator

    if not isinstance(shape, ShapeFunctionOperator):
        raise TypeError("shape must be a ShapeFunctionOperator")
    if not isinstance(material, MaterialStiffnessOperator):
        raise TypeError("material must be a MaterialStiffnessOperator")
    strain = StrainDisplacementOperator(shape, ndof=int(ndof))
    n_elem = 4 * int(ndof)
    contributions = [
        np.zeros((n_elem, n_elem), dtype=np.float64) for _ in range(5)
    ]
    k_membrane, k_coupling, k_bending, k_shear, k_drilling = contributions

    gp3, gw3 = _gauss(3)
    xi3, eta3 = np.meshgrid(gp3, gp3, indexing="ij")
    xi3f, eta3f = xi3.ravel(), eta3.ravel()
    _, _, determinant3, _ = shape.physical_coordinate_form(xi3f, eta3f)
    weights3 = (gw3[:, None] * gw3[None, :]).ravel() * determinant3
    bm, bb, _, _ = strain.natural_coordinate_form(xi3f, eta3f)
    k_membrane += np.einsum(
        "g,gai,ab,gbj->ij", weights3, bm, material.A_mat, bm, optimize=True
    )
    k_coupling += np.einsum(
        "g,gai,ab,gbj->ij", weights3, bm, material.B_mat, bb, optimize=True
    )
    k_coupling += np.einsum(
        "g,gai,ab,gbj->ij",
        weights3,
        bb,
        material.B_mat.T,
        bm,
        optimize=True,
    )
    k_bending += np.einsum(
        "g,gai,ab,gbj->ij", weights3, bb, material.D_mat, bb, optimize=True
    )

    gp2, gw2 = _gauss(2)
    xi2, eta2 = np.meshgrid(gp2, gp2, indexing="ij")
    xi2f, eta2f = xi2.ravel(), eta2.ravel()
    _, _, determinant2, _ = shape.physical_coordinate_form(xi2f, eta2f)
    weights2 = (gw2[:, None] * gw2[None, :]).ravel() * determinant2
    _, _, bs, bd = strain.natural_coordinate_form(xi2f, eta2f)
    k_shear += np.einsum(
        "g,gai,ab,gbj->ij", weights2, bs, material.A_s_mat, bs, optimize=True
    )
    if int(ndof) == 6 and bd is not None:
        k_drilling += material.drilling_stiffness * np.einsum(
            "g,gai,gbj->ij", weights2, bd, bd, optimize=True
        )
    return Mitc4StiffnessContributions(*contributions)


def mitc4_stiffness_contributions(
    L_s: float,
    L_x: float,
    ABD: NDArray,
    thickness: float,
    ks: float = 5.0 / 6.0,
    G_eff: float | None = None,
    include_drilling: bool = True,
    alpha_drill: float = 0.25,
    *,
    A_s: NDArray[np.float64] | None = None,
) -> Mitc4StiffnessContributions:
    """Compatibility constructor exposing the named ``K`` decomposition."""
    from .utilities.D_matrix import MaterialStiffnessOperator
    from .utilities.shape_functions import ShapeFunctionOperator

    shape = ShapeFunctionOperator(float(L_s), float(L_x))
    material = MaterialStiffnessOperator(
        ABD,
        float(thickness),
        ks=float(ks),
        G_eff=G_eff if A_s is None else None,
        alpha_drill=float(alpha_drill),
        A_s=A_s,
    )
    return assemble_mitc4_stiffness(
        shape, material, ndof=6 if include_drilling else 5
    )


def mitc4_stiffness(
    L_s: float,
    L_x: float,
    ABD: NDArray,
    thickness: float,
    ks: float = 5.0 / 6.0,
    G_eff: float | None = None,
    include_drilling: bool = True,
    alpha_drill: float = 0.25,
) -> NDArray:
    """
    MITC4 or MITC4+ element stiffness matrix.

    Parameters
    ----------
    L_s             : element length along contour s [m]
    L_x             : element length along span x [m]
    ABD             : 6x6 laminate stiffness [[A,B],[B,D]]
    thickness       : wall thickness [m]
    ks              : shear correction factor (default 5/6)
    G_eff           : effective transverse shear modulus [Pa]; estimated from A66 if None
    include_drilling: add Hughes-Brezzi drilling DOF (default True) -> 24x24; False -> 20x20
    alpha_drill     : drilling stabilisation factor (Hughes-Brezzi, default 0.25)

    Returns
    -------
    K : (24, 24) default MITC4+; (20, 20) if include_drilling=False (MITC4)
    """
    cache_key = (
        round(float(L_s), 12),
        round(float(L_x), 12),
        _abd_fingerprint(ABD),
        round(float(thickness), 12),
        round(float(ks), 12),
        None if G_eff is None else round(float(G_eff), 6),
        bool(include_drilling),
        round(float(alpha_drill), 12),
    )
    cached = _KE_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return np.asarray(cached)

    k = mitc4_stiffness_contributions(
        L_s,
        L_x,
        ABD,
        thickness,
        ks=ks,
        G_eff=G_eff,
        include_drilling=include_drilling,
        alpha_drill=alpha_drill,
    ).K_e
    _KE_TEMPLATE_CACHE[cache_key] = k.copy()
    return k


# ---------------------------------------------------------------------------
# Resultant recovery
# ---------------------------------------------------------------------------

def mitc4_resultants(
    d_elem: NDArray,
    L_s: float,
    L_x: float,
    ABD: NDArray,
    ndof: int = 6,
) -> dict[str, float]:
    """
    Recover shell resultants at element centroid (xi=0, eta=0).

    Parameters
    ----------
    d_elem : element DOF vector, length 4*ndof (20 for MITC4, 24 for MITC4+)
    ndof   : DOFs per node (5 = MITC4, 6 = MITC4+)

    Returns dict with keys: N11, N22, N12, M11, M22, M12 (N/m or N.m/m).
    """
    Bm = _b_membrane(0.0, 0.0, L_s, L_x, ndof)
    Bb = _b_bending(0.0, 0.0, L_s, L_x, ndof)

    eps0 = Bm @ d_elem
    kappa = Bb @ d_elem

    A_mat = ABD[:3, :3]
    B_mat = ABD[:3, 3:]
    D_mat = ABD[3:, 3:]

    N_vec = A_mat @ eps0 + B_mat @ kappa
    M_vec = B_mat.T @ eps0 + D_mat @ kappa

    return {
        "N11": float(N_vec[0]),
        "N22": float(N_vec[1]),
        "N12": float(N_vec[2]),
        "M11": float(M_vec[0]),
        "M22": float(M_vec[1]),
        "M12": float(M_vec[2]),
    }


def mitc4_resultants_at(
    d_elem: NDArray,
    L_s: float,
    L_x: float,
    ABD: NDArray,
    *,
    xi: float,
    eta: float,
    ndof: int = 6,
) -> dict[str, float]:
    """
    Recover shell resultants at an arbitrary parent-space location (xi, eta).
    """
    Bm = _b_membrane(xi, eta, L_s, L_x, ndof)
    Bb = _b_bending(xi, eta, L_s, L_x, ndof)
    eps0 = Bm @ d_elem
    kappa = Bb @ d_elem
    A_mat = ABD[:3, :3]
    B_mat = ABD[:3, 3:]
    D_mat = ABD[3:, 3:]
    N_vec = A_mat @ eps0 + B_mat @ kappa
    M_vec = B_mat.T @ eps0 + D_mat @ kappa
    return {
        "N11": float(N_vec[0]),
        "N22": float(N_vec[1]),
        "N12": float(N_vec[2]),
        "M11": float(M_vec[0]),
        "M22": float(M_vec[1]),
        "M12": float(M_vec[2]),
    }


def mitc4_geometric_stiffness(
    L_s: float,
    L_x: float,
    N11: float,
    N22: float,
    N12: float,
    *,
    n_gauss: int = 2,
    ndof: int = 6,
) -> NDArray:
    """
    Initial-stress geometric stiffness matrix for a MITC4/MITC4+ element.

    Only the w (out-of-plane) DOF contributes at leading order.
    w DOF is at index 2 within each ndof-block.

    Returns (4*ndof, 4*ndof) matrix.
    """
    n_elem = 4 * ndof
    Kg = np.zeros((n_elem, n_elem))
    w_idx = [I * ndof + 2 for I in range(4)]
    from .utilities.shape_functions import ShapeFunctionOperator

    shape = ShapeFunctionOperator(L_s, L_x)
    gp, gw = _gauss(n_gauss)
    for xi, wi in zip(gp, gw):
        for eta, wj in zip(gp, gw):
            _, gradients, determinant, _ = shape.physical_coordinate_form(
                np.array([xi]), np.array([eta])
            )
            dNdx = gradients[0, :, 0]
            dNds = gradients[0, :, 1]
            w = wi * wj * determinant[0]
            for i in range(4):
                for j in range(4):
                    Kg[w_idx[i], w_idx[j]] += w * (
                        N11  * dNdx[i] * dNdx[j]
                        + N12 * (dNdx[i] * dNds[j] + dNds[i] * dNdx[j])
                        + N22  * dNds[i] * dNds[j]
                    )
    return Kg


def mitc4_shear_resultants(
    d_elem: NDArray,
    L_s: float,
    L_x: float,
    G_eff: float,
    thickness: float,
    *,
    ks: float = 5.0 / 6.0,
    ndof: int = 6,
) -> tuple[float, float]:
    """
    FSDT transverse shear resultants at element centroid (xi=0, eta=0).

    Returns (Q1, Q2) in [N/m].
    """
    Bs = _b_shear_mitc(0.0, 0.0, L_s, L_x, ndof)
    gamma = Bs @ d_elem
    H_s = ks * G_eff * thickness
    return float(H_s * gamma[0]), float(H_s * gamma[1])


def mitc4_generalized_resultants_at(
    d_elem: NDArray,
    L_s: float,
    L_x: float,
    constitutive: object,
    *,
    xi: float = 0.0,
    eta: float = 0.0,
    ndof: int = 6,
) -> dict[str, float]:
    """Recover canonical ``N/M/Q`` from the assembly ABDAs object."""
    from blade_structure_utilities.material_utilities.laminate import (
        FSDTLaminateStiffness,
    )
    from .utilities.B_matrix import StrainDisplacementOperator
    from .utilities.shape_functions import ShapeFunctionOperator

    if not isinstance(constitutive, FSDTLaminateStiffness):
        raise TypeError("constitutive must be FSDTLaminateStiffness")
    b_generalized = StrainDisplacementOperator(
        ShapeFunctionOperator(L_s, L_x), ndof=ndof
    ).generalized(np.array([xi]), np.array([eta]))[0]
    strain = b_generalized @ np.asarray(d_elem, dtype=np.float64)
    resultants = constitutive.C_ABDAs @ strain
    names = ("N11", "N22", "N12", "M11", "M22", "M12", "Q1", "Q2")
    return {name: float(value) for name, value in zip(names, resultants)}


def mitc4_edge_resultants(
    d_elem: NDArray,
    L_s: float,
    L_x: float,
    ABD: NDArray,
    ndof: int = 6,
) -> dict[str, dict[str, float]]:
    """
    Recover membrane resultants at edge midpoints for strip interface diagnostics.

    Returns keys: start (xi=-1, eta=0), end (xi=+1, eta=0).
    """
    start = mitc4_resultants_at(d_elem, L_s, L_x, ABD, xi=-1.0, eta=0.0, ndof=ndof)
    end = mitc4_resultants_at(d_elem, L_s, L_x, ABD, xi=1.0, eta=0.0, ndof=ndof)
    return {
        "start": {"N11": float(start["N11"]), "N12": float(start["N12"])},
        "end": {"N11": float(end["N11"]), "N12": float(end["N12"])},
    }


def mitc4_edge_shear_traction_integrated(
    d_elem: NDArray,
    L_s: float,
    L_x: float,
    ABD: NDArray,
    *,
    edge: str,
    gauss_n: int = 4,
    ndof: int = 6,
) -> dict[str, object]:
    """
    Line-integrate membrane traction components on interface edges (xi=+/-1).

    Returns mean traction over the edge in curvilinear **C** (≡ **E**):
      Nxy_edge_int / Tx_edge_int : x-directed (span / x1) traction
      Ts_edge_int                : s-directed (contour / x2) traction
      Tx_gps, Ts_gps             : per-Gauss-point values
    """
    if edge not in ("start", "end"):
        raise ValueError("edge must be 'start' or 'end'")
    xi_val = -1.0 if edge == "start" else 1.0
    normal_sign = -1.0 if edge == "start" else 1.0
    gp, gw = _gauss(gauss_n)
    j_edge = L_x / 2.0
    tx_int = 0.0
    ts_int = 0.0
    tx_gps: list[float] = []
    ts_gps: list[float] = []
    for eta, w in zip(gp, gw):
        res = mitc4_resultants_at(
            d_elem, L_s, L_x, ABD, xi=xi_val, eta=float(eta), ndof=ndof
        )
        nx = float(res["N11"])
        ny = float(res["N22"])
        nxy = float(res["N12"])
        nvec = np.array([0.0, normal_sign], dtype=float)
        nmat = np.array([[nx, nxy], [nxy, ny]], dtype=float)
        tvec = nmat @ nvec
        tx_gp = float(tvec[0])
        ts_gp = float(tvec[1])
        tx_gps.append(tx_gp)
        ts_gps.append(ts_gp)
        tx_int += float(w) * j_edge * tx_gp
        ts_int += float(w) * j_edge * ts_gp
    edge_len = max(L_x, 1e-30)
    tx_mean = tx_int / edge_len
    ts_mean = ts_int / edge_len
    return {
        "Nxy_edge_int": float(tx_mean),
        "Tx_edge_int": float(tx_mean),
        "Ts_edge_int": float(ts_mean),
        "Tx_gps": tx_gps,
        "Ts_gps": ts_gps,
    }
