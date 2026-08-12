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
  beta_s : Mindlin rotation about contour tangent  (~= -dw/dx in thin limit)
  beta_x : Mindlin rotation about span axis        (~=  dw/ds in thin limit)
  beta_z : drilling rotation about shell normal (MITC4+ only, Hughes-Brezzi stabilisation)

DOF ordering inside the n-vector (n = 4*ndof):
  [node1_ux, node1_us, node1_w, node1_bs, node1_bx, (node1_bz,)
   node2_ux, ..., node4_bx, (node4_bz)]
"""

from __future__ import annotations

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
# Bilinear shape functions
# ---------------------------------------------------------------------------

_XI_NODES = np.array([-1.0, 1.0, 1.0, -1.0])
_ET_NODES = np.array([-1.0, -1.0, 1.0, 1.0])


def _shape(xi: float, eta: float) -> NDArray:
    """N[I] = 1/4*(1 + xi_I*xi)*(1 + eta_I*eta)  for I=0..3."""
    return 0.25 * (1.0 + _XI_NODES * xi) * (1.0 + _ET_NODES * eta)


def _dshape(xi: float, eta: float) -> NDArray:
    """dN/dxi (row 0) and dN/deta (row 1), shape (2, 4)."""
    dNdxi = 0.25 * _XI_NODES * (1.0 + _ET_NODES * eta)
    dNdet = 0.25 * _ET_NODES * (1.0 + _XI_NODES * xi)
    return np.stack([dNdxi, dNdet], axis=0)


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
    dN = _dshape(xi, eta)
    dNds = dN[0] * (2.0 / L_s)
    dNdx = dN[1] * (2.0 / L_x)

    B = np.zeros((3, 4 * ndof))
    for I in range(4):
        base = I * ndof
        B[0, base + 0] = dNdx[I]   # ε11 = du_x/dx
        B[1, base + 1] = dNds[I]   # ε22 = du_s/ds
        B[2, base + 0] = dNds[I]   # γ12 = du_x/ds + du_s/dx
        B[2, base + 1] = dNdx[I]
    return B


def _b_bending(xi: float, eta: float, L_s: float, L_x: float, ndof: int = 6) -> NDArray:
    """
    Bending B matrix (3 x 4*ndof).

    Curvatures: [kappa_xx, kappa_ss, kappa_xs]  from beta_s, beta_x DOFs.
      kappa_xx = d_beta_s/dx,  kappa_ss = -d_beta_x/ds,
      kappa_xs = d_beta_s/ds - d_beta_x/dx
    beta_z (DOF 5, ndof=6) does not contribute to bending curvatures.
    """
    dN = _dshape(xi, eta)
    dNds = dN[0] * (2.0 / L_s)
    dNdx = dN[1] * (2.0 / L_x)

    B = np.zeros((3, 4 * ndof))
    for I in range(4):
        base = I * ndof
        B[0, base + 3] = dNdx[I]   # kappa_xx = d_beta_s/dx
        B[2, base + 3] = dNds[I]   # kappa_xs += d_beta_s/ds
        B[1, base + 4] = -dNds[I]  # kappa_ss = -d_beta_x/ds
        B[2, base + 4] = -dNdx[I]  # kappa_xs += -d_beta_x/dx
    return B


def _b_shear_mitc(xi: float, eta: float, L_s: float, L_x: float, ndof: int = 6) -> NDArray:
    """
    MITC4 mixed-interpolated transverse shear B matrix (2 x 4*ndof).

    Shear strains: [gamma_xn, gamma_sn]  (MITC tying, Bathe & Dvorkin 1985).
    beta_z (DOF 5, ndof=6) does not contribute to transverse shear.
    """
    B = np.zeros((2, 4 * ndof))

    # gamma_xn tying: tying points at eta=+1 (A) and eta=-1 (B)
    for sign_eta, coeff in ((+1.0, 0.5 * (1.0 + eta)), (-1.0, 0.5 * (1.0 - eta))):
        dN_tp = _dshape(xi, sign_eta)
        dNdx_tp = dN_tp[1] * (2.0 / L_x)
        N_tp = _shape(xi, sign_eta)
        for I in range(4):
            base = I * ndof
            B[0, base + 2] += coeff * dNdx_tp[I]   # dw/dx at tying point
            B[0, base + 3] -= coeff * N_tp[I]       # -beta_s at tying point

    # gamma_sn tying: tying points at xi=+1 (C) and xi=-1 (D)
    for sign_xi, coeff in ((+1.0, 0.5 * (1.0 + xi)), (-1.0, 0.5 * (1.0 - xi))):
        dN_tp = _dshape(sign_xi, eta)
        dNds_tp = dN_tp[0] * (2.0 / L_s)
        N_tp = _shape(sign_xi, eta)
        for I in range(4):
            base = I * ndof
            B[1, base + 2] += coeff * dNds_tp[I]   # dw/ds at tying point
            B[1, base + 4] += coeff * N_tp[I]       # +beta_x at tying point

    return B


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
    dN = _dshape(xi, eta)
    dNds = dN[0] * (2.0 / L_s)
    dNdx = dN[1] * (2.0 / L_x)
    N = _shape(xi, eta)

    B = np.zeros((1, 24))
    for I in range(4):
        base = I * 6
        B[0, base + 0] = -dNds[I]   # from -du_x/ds
        B[0, base + 1] = +dNdx[I]   # from +du_s/dx
        B[0, base + 5] = -N[I]      # from -beta_z
    return B


# ---------------------------------------------------------------------------
# Element stiffness
# ---------------------------------------------------------------------------

_KE_TEMPLATE_CACHE: dict[tuple, NDArray] = {}


def _abd_fingerprint(abd: NDArray) -> tuple:
    return tuple(float(x) for x in np.asarray(abd, dtype=np.float64).ravel().round(12))


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

    from .utilities.B_matrix import StrainDisplacementOperator
    from .utilities.D_matrix import MaterialStiffnessOperator
    from .utilities.shape_functions import ShapeFunctionOperator

    ndof = 6 if include_drilling else 5
    shape = ShapeFunctionOperator(float(L_s), float(L_x))
    strain = StrainDisplacementOperator(shape, ndof=ndof)
    material = MaterialStiffnessOperator(
        ABD,
        float(thickness),
        ks=float(ks),
        G_eff=G_eff,
        alpha_drill=float(alpha_drill),
    )
    a_mat, b_mat, d_mat, h_s, c_drill = material.assembly_form()
    d_mb = np.block([[a_mat, b_mat], [b_mat.T, d_mat]])

    jdet = (float(L_s) / 2.0) * (float(L_x) / 2.0)
    n_elem = 4 * ndof
    k = np.zeros((n_elem, n_elem), dtype=np.float64)

    gp3, gw3 = _gauss(3)
    xi3, eta3 = np.meshgrid(gp3, gp3, indexing="ij")
    xi3f = xi3.ravel()
    eta3f = eta3.ravel()
    w3 = (gw3[:, None] * gw3[None, :]).ravel() * jdet
    bm, bb, _, _ = strain.natural_coordinate_form(xi3f, eta3f)
    b_mb = np.concatenate([bm, bb], axis=1)
    k += np.einsum("g,gai,ab,gbj->ij", w3, b_mb, d_mb, b_mb, optimize=True)

    gp2, gw2 = _gauss(2)
    xi2, eta2 = np.meshgrid(gp2, gp2, indexing="ij")
    xi2f = xi2.ravel()
    eta2f = eta2.ravel()
    w2 = (gw2[:, None] * gw2[None, :]).ravel() * jdet
    _, _, bs, bd = strain.natural_coordinate_form(xi2f, eta2f)
    h2 = np.array([[h_s, 0.0], [0.0, h_s]], dtype=np.float64)
    k += np.einsum("g,gai,ab,gbj->ij", w2, bs, h2, bs, optimize=True)
    if include_drilling and bd is not None:
        k += c_drill * np.einsum("g,gai,gbj->ij", w2, bd, bd, optimize=True)

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
    Jdet = (L_x / 2.0) * (L_s / 2.0)
    gp, gw = _gauss(n_gauss)
    for xi, wi in zip(gp, gw):
        for eta, wj in zip(gp, gw):
            dN = _dshape(xi, eta)
            dNdx = dN[1] * (2.0 / L_x)
            dNds = dN[0] * (2.0 / L_s)
            w = wi * wj * Jdet
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
