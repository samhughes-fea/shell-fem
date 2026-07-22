"""
Cross-sectional stiffness homogenisation from the MITC4/CLPT shell model.

K6 is extracted via the energy bilinear form K6[m,n] = U_m^T @ K_global @ U_n,
where U_m is the unit-mode displacement field for beam mode m applied across a
unit-length (L_x = 1 m) shell slice, and K_global is the unconstrained MITC4
stiffness from ``assemble_global_shell``.

The 6 Saint-Venant beam modes and their DOF prescriptions on outboard cut-face
nodes (relative to the inboard reference via periodic jumps):

  Mode 0  ε₀      : u_x jump = L_x           (uniform axial extension)
  Mode 1  κ_y     : u_x jump = -(z - z_e)·L_x (bending about y-axis)
  Mode 2  κ_z     : u_x jump = +(y - y_e)·L_x (bending about z-axis)
  Mode 3  γ_t     : torsion / twist jump
  Mode 4  γ_s_y   : transverse shear y
  Mode 5  γ_s_z   : transverse shear z

Neither cut face is clamped — this is an RVE / unit-cell homogenisation along
the beam axis. A 7th warping/bimoment mode is permanently out of scope.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from blade_precompute.section_shell_model.pre_processing.containers.fem_config import MITC4_DEFAULTS


_NDOF_NODE = MITC4_DEFAULTS.n_dof_node
_U_X = 0
_U_S = 1
_W = 2
_BETA_S = 3
_BETA_X = 4
_BETA_ZET = 5   # drilling rotation (MITC4+)


# ---------------------------------------------------------------------------
# Section centroid helpers
# ---------------------------------------------------------------------------

def compute_elastic_centroid_from_panels(panels: list[Any]) -> tuple[float, float]:
    """A11-weighted elastic centroid (y_e, z_e) from panel laminates and geometry.

    Uses the A₁₁ entry of each panel's ABD matrix as the axial stiffness weight,
    consistent with the strip-solver convention.
    """
    from ..pre_processing.constitutive.panel_abd import panel_abd_tuple_from_lam

    sum_A11_y = 0.0
    sum_A11_z = 0.0
    sum_A11 = 0.0
    for p in panels:
        try:
            abd, _th, _nu, _E, _G = panel_abd_tuple_from_lam(p.lam)
            A11 = float(abd[0, 0])
        except Exception as exc:
            warnings.warn(
                "elastic-centroid: skipping panel "
                f"{getattr(p, 'label', '<unknown>')!r} — ABD assembly failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        s_p = np.asarray(p.s, dtype=float)
        if len(s_p) < 2 or len(nodes_yz) < 2:
            continue
        L_p = float(abs(s_p[-1] - s_p[0]))
        y_mid = float(np.mean(nodes_yz[:, 0]))
        z_mid = float(np.mean(nodes_yz[:, 1]))
        sum_A11_y += A11 * L_p * y_mid
        sum_A11_z += A11 * L_p * z_mid
        sum_A11 += A11 * L_p

    if sum_A11 < 1e-30:
        return 0.0, 0.0
    return float(sum_A11_y / sum_A11), float(sum_A11_z / sum_A11)


# ---------------------------------------------------------------------------
# Unit-mode displacement matrix
# ---------------------------------------------------------------------------

def _build_unit_mode_matrix(
    node_meta: Any,
    y_e: float,
    z_e: float,
    omega_hat: NDArray | None,
    *,
    ndof: int = _NDOF_NODE,
) -> NDArray[np.float64]:
    """Build (n_gdof, 7) matrix U of unit beam-mode displacement fields.

    Only top-layer nodes (``node_meta.is_top == True``) carry non-zero entries.
    Bottom-layer nodes represent the clamped reference face (zero displacement).
    MPC virtual nodes with NaN coordinates are skipped.

    ``ndof`` is the per-node DOF count (6 for MITC4+, 8 for HSDT).  Only the
    shared u_x/u_s slots (indices 0/1) are populated, so any extra higher-order
    DOFs are left at zero in the prescribed modes.
    """
    n_nodes = node_meta.n_nodes
    n_gdof = n_nodes * ndof
    U = np.zeros((n_gdof, 7), dtype=np.float64)

    yz = node_meta.yz
    is_top = node_meta.is_top
    tan = node_meta.tangent_yz

    for gn in range(n_nodes):
        if not is_top[gn]:
            continue
        if np.isnan(yz[gn, 0]):
            continue  # virtual MPC cluster node — no direct physical coords

        y_i = float(yz[gn, 0])
        z_i = float(yz[gn, 1])
        ty_i = float(tan[gn, 0]) if not np.isnan(tan[gn, 0]) else 0.0
        tz_i = float(tan[gn, 1]) if not np.isnan(tan[gn, 1]) else 0.0
        base = gn * ndof

        # Axial DOF (u_x) modes
        U[base + _U_X, 0] = 1.0             # ε₀: uniform axial
        U[base + _U_X, 1] = -(z_i - z_e)   # κ_y: bending about y
        U[base + _U_X, 2] = (y_i - y_e)    # κ_z: bending about z
        if omega_hat is not None and not np.isnan(omega_hat[gn]):
            U[base + _U_X, 6] = float(omega_hat[gn])  # ψ': warping

        # Tangential shear DOF (u_s) modes
        U[base + _U_S, 3] = 1.0    # γ_t: uniform torsion shear
        U[base + _U_S, 4] = tz_i   # γ_s_y: shear-y (tz component of tangent)
        U[base + _U_S, 5] = -ty_i  # γ_s_z: shear-z (-ty component)

    return U


# ---------------------------------------------------------------------------
# Periodic unit-cell BC — correct K6 without Saint-Venant end effects
# ---------------------------------------------------------------------------


def _build_node_pairs(node_meta: Any) -> list[tuple[int, int]]:
    """Return [(i_inboard, i_outboard), ...] matched by (y, z) coordinate.

    i_inboard:  is_top=False node at x=0 (inboard cut face)
    i_outboard: is_top=True  node at x=L_x (outboard cut face)
    """
    from scipy.spatial import KDTree

    yz = node_meta.yz          # (n_nodes, 2)
    is_top = node_meta.is_top  # (n_nodes,) bool

    physical = ~np.isnan(yz[:, 0])
    idx_all = np.where(physical)[0]

    inboard_idx = idx_all[~is_top[idx_all]]
    outboard_idx = idx_all[is_top[idx_all]]

    assert len(inboard_idx) == len(outboard_idx), (
        f"Inboard/outboard node count mismatch: {len(inboard_idx)} vs {len(outboard_idx)}"
    )

    tree = KDTree(yz[inboard_idx])
    dist, loc = tree.query(yz[outboard_idx])
    assert np.max(dist) < 1e-9, f"Node pairing failed: max dist = {np.max(dist):.3e}"

    return [(int(inboard_idx[loc[k]]), int(outboard_idx[k])) for k in range(len(outboard_idx))]


def _build_node_pairs_topological(
    panel_maps: list[Any],
    node_meta: Any,
) -> list[tuple[int, int]]:
    """Return panel-local inboard/outboard node pairs from solver topology."""
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pm in panel_maps:
        n = len(getattr(pm, "s_nodes", []))
        if n <= 0:
            continue
        global_nodes = list(getattr(pm, "global_nodes", []))
        if len(global_nodes) != 2 * n:
            raise ValueError(
                f"Panel map '{getattr(pm, 'panel_label', '?')}' has invalid node topology: "
                f"expected {2 * n} global nodes, got {len(global_nodes)}"
            )
        for i in range(n):
            pair = (int(global_nodes[i]), int(global_nodes[i + n]))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)

    if not pairs:
        return []

    yz = np.asarray(node_meta.yz, dtype=float)
    is_top = np.asarray(node_meta.is_top, dtype=bool)
    for i_in, i_out in pairs:
        if is_top[i_in] or not is_top[i_out]:
            raise ValueError("Topological periodic pairing found invalid inboard/outboard node ordering")
        if np.isnan(yz[i_in, 0]) or np.isnan(yz[i_out, 0]):
            raise ValueError("Topological periodic pairing encountered non-physical node metadata")
    return pairs


def _build_node_pairs_by_group(node_meta: Any) -> list[tuple[int, int]]:
    """Return periodic node pairs from physical pair-group IDs carried in node metadata."""
    yz = np.asarray(node_meta.yz, dtype=float)
    is_top = np.asarray(node_meta.is_top, dtype=bool)
    pair_group = np.asarray(getattr(node_meta, "pair_group", np.full(len(yz), -1, dtype=np.int32)), dtype=np.int32)

    pairs_by_group: dict[int, dict[str, list[int]]] = {}
    for gn, gid in enumerate(pair_group):
        if gid < 0 or np.isnan(yz[gn, 0]):
            continue
        slot = "top" if bool(is_top[gn]) else "bottom"
        pairs_by_group.setdefault(int(gid), {"bottom": [], "top": []})[slot].append(int(gn))

    pairs: list[tuple[int, int]] = []
    for gid in sorted(pairs_by_group):
        bottom = sorted(set(pairs_by_group[gid]["bottom"]))
        top = sorted(set(pairs_by_group[gid]["top"]))
        if len(bottom) != 1 or len(top) != 1:
            raise ValueError(
                f"Periodic pair group {gid} is ambiguous: bottom={bottom}, top={top}"
            )
        pairs.append((bottom[0], top[0]))

    return pairs


def _macro_jump(
    yz_out: Any, tan_out: Any, y_e: float, z_e: float, L_x: float
) -> tuple[np.ndarray, np.ndarray]:
    """Prescribed inboard→outboard displacement jump for the 6 beam unit modes."""
    y_i, z_i = float(yz_out[0]), float(yz_out[1])
    ty_i = float(tan_out[0]) if tan_out is not None else 0.0
    tz_i = float(tan_out[1]) if tan_out is not None else 1.0
    # mode: 0(ε₀) 1(κ_y) 2(κ_z) 3(γ_t) 4(γ_z) 5(γ_y)
    delta_ux = np.array([L_x, -(z_i - z_e) * L_x, (y_i - y_e) * L_x, 0.0, 0.0, 0.0])
    delta_us = np.array([0.0, 0.0, 0.0, L_x, tz_i * L_x, -ty_i * L_x])
    return delta_ux, delta_us


def _macro_jump_dofs(
    delta_ux: np.ndarray,
    delta_us: np.ndarray,
    ty: float,
    tz: float,
    dof_frame: str,
) -> dict[int, np.ndarray]:
    """Map the (axial, tangential) macro jumps onto per-node local DOF indices.

    ``dof_frame="local"`` (MITC4 / HSDT): the in-plane translation DOFs are
    ``[u_x (span), u_s (contour-tangential), w (normal)]`` — the axial jump goes on
    ``_U_X`` and the tangential jump on ``_U_S``.

    ``dof_frame="global"`` (CalculiX S4R): the translation DOFs are global
    ``[u_x (span), u_y, u_z]`` — the axial jump still goes on index 0 (global x = span),
    but the tangential jump ``u_s`` is resolved onto the global ``(u_y, u_z)`` components
    via the node tangent ``(t_y, t_z)``.
    """
    if dof_frame == "global":
        return {0: delta_ux, 1: ty * delta_us, 2: tz * delta_us}
    if dof_frame != "local":
        raise ValueError(f"dof_frame must be 'local' or 'global'; got {dof_frame!r}")
    return {_U_X: delta_ux, _U_S: delta_us}


def _build_periodic_transform_simple(
    pairs: list[tuple[int, int]],
    n_gdof: int,
    ndof: int,
    yz: np.ndarray,
    tangents: Any,
    y_e: float,
    z_e: float,
    L_x: float,
    *,
    dof_frame: str = "local",
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Periodic transform for shared/independent topology (every node periodic-paired).

    Free columns = inboard cut-face DOFs except the reference node's; the outboard
    twin equals its inboard partner plus the macro-mode jump ``d_0``. ``dof_frame``
    selects how the macro jump maps onto node DOFs (see :func:`_macro_jump_dofs`):
    ``"local"`` for MITC4/HSDT (contour-local ``u_s``), ``"global"`` for CalculiX S4R
    (global ``u_y/u_z``).
    """
    ref_inboard = pairs[0][0]
    col_map = np.full(n_gdof, -1, dtype=np.int64)
    col = 0
    for i_in, _i_out in pairs:
        if i_in == ref_inboard:
            continue
        for k in range(ndof):
            col_map[ndof * i_in + k] = col
            col += 1
    n_free = col

    rows_T: list[int] = []
    cols_T: list[int] = []
    vals_T: list[float] = []
    d_0 = np.zeros((n_gdof, 6))
    for i_in, i_out in pairs:
        tan_o = tangents[i_out] if tangents is not None else None
        delta_ux, delta_us = _macro_jump(yz[i_out], tan_o, y_e, z_e, L_x)
        ty_o = float(tan_o[0]) if tan_o is not None else 0.0
        tz_o = float(tan_o[1]) if tan_o is not None else 1.0
        jump_map = _macro_jump_dofs(delta_ux, delta_us, ty_o, tz_o, dof_frame)
        for k in range(ndof):
            g_in = ndof * i_in + k
            g_out = ndof * i_out + k
            c = col_map[g_in]
            if c >= 0:
                rows_T.append(g_in); cols_T.append(int(c)); vals_T.append(1.0)
                rows_T.append(g_out); cols_T.append(int(c)); vals_T.append(1.0)
            if k in jump_map:
                d_0[g_out, :] = jump_map[k]
    T = sp.csr_matrix((vals_T, (rows_T, cols_T)), shape=(n_gdof, n_free))
    return T, d_0


def _build_periodic_transform_clustered(
    pairs: list[tuple[int, int]],
    constraints: dict[int, list[tuple[int, float]]],
    n_gdof: int,
    ndof: int,
    yz: np.ndarray,
    tangents: Any,
    y_e: float,
    z_e: float,
    L_x: float,
    clamp_dofs: set[int] | None = None,
    *,
    dof_frame: str = "local",
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Periodic transform composing cluster-basis MPCs with periodic pairing.

    The cluster-basis (``transformed_basis``) interface ties panel endpoint DOFs to
    virtual cluster reference DOFs via ``constraints`` (slave -> sum coeff*master).
    The periodic homogeniser must apply **both** the junction-compatibility MPCs and
    the inboard/outboard periodic pairing, otherwise the cluster master DOFs are
    unconstrained (singular) and the junction is kinematically disconnected.

    Independent columns = inboard MASTER DOFs (not a cluster slave, not the reference
    node). Cluster slaves are expressed via their MPC; outboard master DOFs equal
    their inboard twin plus the macro jump; outboard slave jumps are propagated from
    their (outboard) masters. ``clamp_dofs`` is a set of global DOFs pinned to zero
    (genuinely unsupported virtual DOFs), excluded from the free columns.
    """
    slaves = set(constraints)
    clamp = clamp_dofs or set()
    in_nodes = {i_in for i_in, _ in pairs}
    out2in = {i_out: i_in for i_in, i_out in pairs}

    # Reference node: first inboard node that is a true master (no slave DOFs).
    ref_inboard = pairs[0][0]
    for i_in, _i_out in pairs:
        if not any((ndof * i_in + k) in slaves for k in range(ndof)):
            ref_inboard = i_in
            break

    col_map = np.full(n_gdof, -1, dtype=np.int64)
    col_to_gdof: list[int] = []
    col = 0
    for i_in, _i_out in pairs:
        if i_in == ref_inboard:
            continue
        for k in range(ndof):
            g_in = ndof * i_in + k
            if g_in in slaves or g_in in clamp:
                continue
            col_map[g_in] = col
            col_to_gdof.append(g_in)
            col += 1
    n_free = col

    # Macro jump on outboard master DOFs, then propagate to outboard slaves via MPC.
    d_0 = np.zeros((n_gdof, 6))
    for i_in, i_out in pairs:
        tan_o = tangents[i_out] if tangents is not None else None
        delta_ux, delta_us = _macro_jump(yz[i_out], tan_o, y_e, z_e, L_x)
        ty_o = float(tan_o[0]) if tan_o is not None else 0.0
        tz_o = float(tan_o[1]) if tan_o is not None else 1.0
        jump_map = _macro_jump_dofs(delta_ux, delta_us, ty_o, tz_o, dof_frame)
        for k in range(ndof):
            g_out = ndof * i_out + k
            if g_out in slaves:
                continue
            if k in jump_map:
                d_0[g_out, :] = jump_map[k]
    for slave, terms in constraints.items():
        acc = np.zeros(6)
        for m, coeff in terms:
            acc += coeff * d_0[m, :]
        d_0[slave, :] = acc

    rows_T: list[int] = []
    cols_T: list[int] = []
    vals_T: list[float] = []
    # Master DOFs: identity (inboard) + periodic twin (outboard = inboard + jump).
    for i_in, i_out in pairs:
        for k in range(ndof):
            g_in = ndof * i_in + k
            c = col_map[g_in]
            if c >= 0:
                rows_T.append(g_in); cols_T.append(int(c)); vals_T.append(1.0)
                rows_T.append(ndof * i_out + k); cols_T.append(int(c)); vals_T.append(1.0)
    # Slave DOFs: d[slave] = sum coeff*d[master]; map outboard masters to inboard twin column.
    for slave, terms in constraints.items():
        for m, coeff in terms:
            node_m, k_m = divmod(m, ndof)
            if node_m in in_nodes:
                c = col_map[m]
            elif node_m in out2in:
                c = col_map[ndof * out2in[node_m] + k_m]
            else:
                c = -1
            if c >= 0:
                rows_T.append(slave); cols_T.append(int(c)); vals_T.append(float(coeff))
    T = sp.csr_matrix((vals_T, (rows_T, cols_T)), shape=(n_gdof, n_free))
    return T, d_0, np.asarray(col_to_gdof, dtype=np.int64)


def compute_k6_periodic_bc(
    K_global: sp.spmatrix,
    node_meta: Any,
    panel_maps: list[Any],
    panels: list[Any],
    y_e: float,
    z_e: float,
    L_x: float = 1.0,
    *,
    topological: bool = False,
    y_e_out: float | None = None,
    z_e_out: float | None = None,
    ndof: int = _NDOF_NODE,
    dof_frame: str = "local",
    return_bvp: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Any]:
    """Return K6 (6×6, station frame) and u_modes (n_gdof×6) via periodic unit-cell BC.

    Outboard cut-face DOFs = inboard cut-face DOFs + prescribed displacement jump
    encoding each of the 6 beam unit modes (ε₀, κ_y, κ_z, γ_t, γ_z, γ_y).
    Neither cut face is clamped; K6 is obtained from the energy bilinear form.
    All quantities in station frame (S): y = chordwise, z = chord-normal.

    ``dof_frame`` selects how the macro jump maps onto node translation DOFs:
    ``"local"`` (default) for the contour-local convention of MITC4/HSDT, ``"global"``
    for CalculiX S4R whose translation DOFs are global ``[u_x, u_y, u_z]``.

    When ``return_bvp=True``, also returns the :class:`PeriodicReducedBVP` used
    for the factorised free set (shared with face-traction solves).
    """
    from .periodic_reduced_bvp import build_periodic_reduced_bvp

    del panels  # kept for API compatibility with call sites
    bvp = build_periodic_reduced_bvp(
        K_global,
        node_meta,
        panel_maps,
        y_e=y_e,
        z_e=z_e,
        L_x=L_x,
        topological=topological,
        y_e_out=y_e_out,
        z_e_out=z_e_out,
        ndof=ndof,
        dof_frame=dof_frame,
    )
    K6, u_modes = bvp.compute_k6_modes(K_global)
    if return_bvp:
        return K6, u_modes, bvp
    return K6, u_modes
