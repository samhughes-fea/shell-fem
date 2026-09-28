"""Rigid-offset cap-tie MPC equations and chained-master expansion."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.analysis_dof_layout import (
    BETA_S,
    BETA_X,
    BETA_ZET,
    U_S,
    U_X,
    W,
    gdof,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.panel_node_assembly import (
    SectionNodeAssemblyMetadata,
    PanelNodeAssemblyMap,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.cap_tie.geometry import (
    CAP_TIE_MODES,
    CapTieReport,
    build_cap_skin_ties,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.types import (
    Constraints,
)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2-D cross product ``a_y b_z - a_z b_y`` (the x-component of ``a x b``)."""
    return float(a[0] * b[1] - a[1] * b[0])


def _coalesce_terms(terms: list[tuple[int, float]]) -> list[tuple[int, float]]:
    acc: dict[int, float] = {}
    for m, c in terms:
        acc[int(m)] = acc.get(int(m), 0.0) + float(c)
    return [(m, c) for m, c in acc.items() if abs(c) > 1e-14]


def _expand_master_terms(
    terms: list[tuple[int, float]],
    existing: Constraints,
    *,
    _depth: int = 0,
) -> list[tuple[int, float]]:
    """Substitute slave masters through ``existing`` until all masters are free."""
    if _depth > 12:
        raise ValueError(
            "cap-tie master expansion exceeded depth 12 (cyclic constraints?)"
        )
    out: list[tuple[int, float]] = []
    for m, c in terms:
        m_i = int(m)
        if m_i in existing:
            nested = [
                (int(m2), float(c) * float(c2)) for m2, c2 in existing[m_i]
            ]
            out.extend(_expand_master_terms(nested, existing, _depth=_depth + 1))
        else:
            out.append((m_i, float(c)))
    return _coalesce_terms(out)


def cap_tie_constraints(
    report: CapTieReport,
    node_meta: SectionNodeAssemblyMetadata,
    *,
    ndof: int,
    existing: Constraints | None = None,
) -> Constraints:
    """Turn located ties into rigid-link MPCs on the six shared DOFs.

    Cap nodes already constrained by junction MPCs are skipped. Skin masters
    that are themselves slaves are expanded through ``existing`` so emitted
    masters stay independent.
    """
    existing = existing or {}
    existing_slaves = set(existing.keys())
    constraints: Constraints = {}
    has_drilling = int(ndof) >= 6
    skipped_junction = 0

    tangents = node_meta.tangent_yz
    for lk in report.links:
        t_c = np.asarray(tangents[lk.slave_node], dtype=float)
        if not np.all(np.isfinite(t_c)):
            raise ValueError(
                f"cap node {lk.slave_node} ({lk.cap_label}) has no tangent in "
                "node_meta; cannot orient a rigid link"
            )
        n_c = np.array([-t_c[1], t_c[0]], dtype=float)
        r = np.asarray(lk.offset_yz, dtype=float)

        slave_dofs = {
            U_X: gdof(lk.slave_node, U_X, ndof),
            U_S: gdof(lk.slave_node, U_S, ndof),
            W: gdof(lk.slave_node, W, ndof),
            BETA_S: gdof(lk.slave_node, BETA_S, ndof),
            BETA_X: gdof(lk.slave_node, BETA_X, ndof),
        }
        if has_drilling:
            slave_dofs[BETA_ZET] = gdof(lk.slave_node, BETA_ZET, ndof)
        if any(d in existing_slaves for d in slave_dofs.values()):
            skipped_junction += 1
            continue
        clash = [d for d in slave_dofs.values() if d in constraints]
        if clash:
            raise ValueError(
                f"cap node {lk.slave_node} ({lk.cap_label}) is already a "
                f"constraint slave on DOFs {sorted(clash)}; refusing to add a "
                "second, conflicting rule"
            )

        terms: dict[int, list[tuple[int, float]]] = {d: [] for d in slave_dofs}
        for gm, n_k in zip(lk.master_nodes, lk.weights):
            if n_k == 0.0:
                continue
            t_k = np.asarray(tangents[gm], dtype=float)
            if not np.all(np.isfinite(t_k)):
                raise ValueError(
                    f"skin master node {gm} ({lk.skin_label}) has no tangent in "
                    "node_meta; cannot orient a rigid link"
                )
            n_kv = np.array([-t_k[1], t_k[0]], dtype=float)
            tt = float(np.dot(t_k, t_c))
            tn = float(np.dot(t_k, n_c))
            nt = float(np.dot(n_kv, t_c))
            nn = float(np.dot(n_kv, n_c))
            cr_t = _cross2(t_k, r)
            cr_n = _cross2(n_kv, r)
            cr_rt = _cross2(r, t_c)
            cr_rn = _cross2(r, n_c)

            m_ux = gdof(gm, U_X, ndof)
            m_us = gdof(gm, U_S, ndof)
            m_w = gdof(gm, W, ndof)
            m_bs = gdof(gm, BETA_S, ndof)
            m_bx = gdof(gm, BETA_X, ndof)
            m_bz = gdof(gm, BETA_ZET, ndof) if has_drilling else None

            terms[U_X] += [(m_ux, n_k), (m_bs, n_k * cr_t)]
            terms[U_S] += [(m_us, n_k * tt), (m_w, n_k * nt), (m_bx, n_k * cr_rt)]
            terms[W] += [(m_us, n_k * tn), (m_w, n_k * nn), (m_bx, n_k * cr_rn)]
            terms[BETA_S] += [(m_bs, n_k * tt)]
            terms[BETA_X] += [(m_bx, n_k)]
            if has_drilling:
                terms[U_X] += [(m_bz, n_k * cr_n)]
                terms[BETA_S] += [(m_bz, n_k * nt)]
                terms[BETA_ZET] += [(m_bs, n_k * tn), (m_bz, n_k * nn)]

        for local, sdof in slave_dofs.items():
            kept = _expand_master_terms(
                [(m, c) for m, c in terms[local] if abs(c) > 1e-14],
                existing,
            )
            if kept:
                constraints[sdof] = kept

    if skipped_junction:
        report.notes.append(
            f"skipped {skipped_junction} cap-tie link(s) already owned by "
            "junction / existing MPCs (transformed_basis endpoints)"
        )
    return constraints


def build_cap_tie_constraints(
    panel_maps: list[PanelNodeAssemblyMap],
    node_meta: SectionNodeAssemblyMetadata,
    *,
    ndof: int,
    mode: str = "none",
    existing: Constraints | None = None,
    shell_midlines: list | tuple | None = None,
) -> tuple[Constraints, CapTieReport]:
    """Locate and emit cap ties in one call. ``mode='none'`` is a no-op.

    ``shell_midlines`` must carry SG-017 ``StripBond`` entries when
    ``mode='skin_offset'`` (FEA-CT-001).
    """
    mode = str(mode or "none")
    if mode not in CAP_TIE_MODES:
        raise ValueError(f"cap_tie_mode must be one of {CAP_TIE_MODES}; got {mode!r}")
    if mode == "none":
        return {}, CapTieReport(mode="none", notes=["cap ties disabled"])
    report = build_cap_skin_ties(
        panel_maps, node_meta, shell_midlines=shell_midlines
    )
    return cap_tie_constraints(report, node_meta, ndof=ndof, existing=existing), report


__all__ = [
    "build_cap_tie_constraints",
    "cap_tie_constraints",
]
