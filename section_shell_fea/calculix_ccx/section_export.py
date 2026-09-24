"""Export full-section strip meshes to CalculiX input decks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.global_numbering import (
    DEFAULT_BC_MODE,
    is_contour_support,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import interp_yz_nodes
from .mesh import CcxSectionMesh, _sanitize_elset_name

_U_X, _U_Y, _U_Z = 0, 1, 2
_TH_X, _TH_Y, _TH_Z = 3, 4, 5


def _support_node_ccx_ids(
    mesh: CcxSectionMesh,
    *,
    bc_mode: str = DEFAULT_BC_MODE,
) -> tuple[list[tuple[int, int, int]], list[int]]:
    """Return boundary cards (nid, dof_start, dof_end) and anchor node ids."""
    contour = is_contour_support(bc_mode)
    support_dofs = (_U_Z, _TH_X, _TH_Y) if contour else (_U_Z, _TH_Y)
    boundary: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()

    for pm in mesh.panel_maps:
        if len(pm.s_nodes) == 0:
            continue
        n = len(pm.s_nodes)
        support_local = (0, n - 1, n, 2 * n - 1) if contour else (0, n - 1)
        for ln in support_local:
            gn = pm.global_nodes[ln]
            ccx_nid = mesh.global_to_ccx_node.get(gn)
            if ccx_nid is None:
                continue
            for d in support_dofs:
                key = (ccx_nid, d + 1)
                if key in seen:
                    continue
                seen.add(key)
                ccx_d = d + 1
                boundary.append((ccx_nid, ccx_d, ccx_d))

    anchor_ids = sorted(mesh.nodes.keys())[:2]
    for ccx_nid in anchor_ids:
        for d in (_U_X, _U_Y):
            key = (ccx_nid, d + 1)
            if key in seen:
                continue
            seen.add(key)
            ccx_d = d + 1
            boundary.append((ccx_nid, ccx_d, ccx_d))

    return boundary, anchor_ids


def _cload_entries(
    mesh: CcxSectionMesh,
    zones: list[Any],
    nx_panels: list[np.ndarray],
    nxy_panels: list[np.ndarray],
) -> list[tuple[int, int, float]]:
    """Nodal loads on top-layer nodes (CalculiX DOF 1=x, 2=y, 3=z)."""
    loads: dict[tuple[int, int], float] = {}
    l_x = mesh.l_x

    for pi, pm in enumerate(mesh.panel_maps):
        if len(pm.s_nodes) == 0:
            continue
        p = zones[pi]
        yz_nodes = interp_yz_nodes(pm.s_nodes, p)
        s_nodes = pm.s_nodes
        nx_nodes = np.interp(
            s_nodes, np.asarray(p.s, dtype=float), np.asarray(nx_panels[pi], dtype=float)
        )
        nxy_nodes = np.interp(
            s_nodes, np.asarray(p.s, dtype=float), np.asarray(nxy_panels[pi], dtype=float)
        )

        for e in pm.elements:
            i0, i1, i2, i3 = e
            l_s = float(abs(s_nodes[i1] - s_nodes[i0]))
            if l_s < 1e-30:
                continue
            nx_avg = 0.5 * (nx_nodes[i0] + nx_nodes[i1])
            nxy_avg = 0.5 * (nxy_nodes[i0] + nxy_nodes[i1])
            ty = float(yz_nodes[i1, 0] - yz_nodes[i0, 0]) / max(l_s, 1e-30)
            tz = float(yz_nodes[i1, 1] - yz_nodes[i0, 1]) / max(l_s, 1e-30)
            f_x = nx_avg * l_s / 2.0
            f_y = nxy_avg * ty * l_s / 2.0
            f_z = nxy_avg * tz * l_s / 2.0
            for ln in (i2, i3):
                gn = pm.global_nodes[ln]
                ccx_nid = mesh.global_to_ccx_node.get(gn)
                if ccx_nid is None:
                    continue
                for dof, fval in ((1, f_x), (2, f_y), (3, f_z)):
                    key = (ccx_nid, dof)
                    loads[key] = loads.get(key, 0.0) + fval

    return [(nid, dof, val) for (nid, dof), val in sorted(loads.items()) if abs(val) > 0.0]


def _chunk_ids(ids: list[int] | tuple[int, ...], width: int = 16) -> list[str]:
    rows: list[str] = []
    for i in range(0, len(ids), width):
        rows.append(", ".join(str(x) for x in ids[i : i + width]))
    return rows


def _emit_cap_skin_ties(
    lines: list[str],
    mesh: CcxSectionMesh,
    *,
    cap_tie_export: str = "tie",
) -> None:
    """Emit bonded cap–skin joints (``*TIE`` or closest-node ``*EQUATION`` fallback).

    CalculiX documentation states ``*TIE`` is for 3-D continuum elements; shells are
    expanded to bricks internally and edge/face ties often work in practice. When a
    CCX build rejects shell ``*TIE``, callers may pass ``cap_tie_export="equation"``
    for translation+rotation locks between each slave node and its nearest master
    node (offset lever-arm kinematics are then approximate).
    """
    ties = list(getattr(mesh, "cap_skin_ties", None) or [])
    if not ties or str(getattr(mesh, "cap_tie_mode", "none")) != "skin_offset":
        return

    export = str(cap_tie_export or "tie").strip().lower()
    if export not in ("tie", "equation"):
        raise ValueError(f"cap_tie_export must be 'tie' or 'equation', got {cap_tie_export!r}")

    for tie in ties:
        slave_nset = f"CapSkinSlave_{tie.name}"
        master_elset = f"CapSkinMasterEls_{tie.name}"
        slave_surf = f"CapSkinSlave_{tie.name}"
        master_surf = f"CapSkinMaster_{tie.name}"

        lines.append(f"*NSET, NSET={slave_nset}")
        lines.extend(_chunk_ids(tie.slave_ccx_nodes))
        lines.append(f"*ELSET, ELSET={master_elset}")
        lines.extend(_chunk_ids(tie.master_ccx_elements))

        if export == "tie":
            lines.append(f"*SURFACE, NAME={slave_surf}, TYPE=NODE")
            lines.append(slave_nset)
            # Both expanded shell faces; POSITION TOLERANCE covers the wall offset.
            lines.append(f"*SURFACE, NAME={master_surf}, TYPE=ELEMENT")
            lines.append(f"{master_elset}, S1")
            lines.append(f"{master_elset}, S2")
            lines.append(
                f"*TIE, NAME=CapSkin_{tie.name}, "
                f"POSITION TOLERANCE={float(tie.position_tolerance):.9e}, ADJUST=YES"
            )
            lines.append(f"{slave_surf}, {master_surf}")
            continue

        # Fallback: equate all 6 DOFs of each slave node to the nearest master node.
        master_nodes: list[int] = []
        for eid in tie.master_ccx_elements:
            for e in mesh.elements:
                if e.eid == eid:
                    master_nodes.extend(int(n) for n in e.node_ids)
                    break
        master_nodes = sorted(set(master_nodes))
        if not master_nodes:
            continue
        for sn in tie.slave_ccx_nodes:
            sx, sy, sz = mesh.nodes[int(sn)]
            best_m, best_d = master_nodes[0], float("inf")
            for mn in master_nodes:
                mx, my, mz = mesh.nodes[int(mn)]
                d = (sy - my) ** 2 + (sz - mz) ** 2 + (sx - mx) ** 2
                if d < best_d:
                    best_d, best_m = d, int(mn)
            for dof in (1, 2, 3, 4, 5, 6):
                _emit_equation(
                    lines,
                    [(int(sn), dof, 1.0), (best_m, dof, -1.0)],
                )


def write_section_inp(
    mesh: CcxSectionMesh,
    zones: list[Any],
    nx_panels: list[np.ndarray],
    nxy_panels: list[np.ndarray],
    path: str | Path,
    *,
    bc_mode: str = DEFAULT_BC_MODE,
    job_name: str | None = None,
    cap_tie_mode: str | None = None,
    cap_tie_export: str = "tie",
) -> Path:
    """Write a full-section S4R CalculiX deck.

    Cap–skin bonds follow MITC4 ``cap_tie_mode``:
    - ``skin_offset`` (default when ``mesh.cap_skin_ties`` is populated): emit ``*TIE``
    - ``none``: omit bonded interface cards

    ``cap_tie_export`` selects ``*TIE`` (default) or closest-node ``*EQUATION`` fallback.
    Skin–web / cap–web junctions remain shared nodes in the mesh topology.
    """
    out = Path(path)
    _ = job_name or out.stem
    # Honour explicit override without rebuilding the mesh.
    if cap_tie_mode is not None:
        mesh.cap_tie_mode = str(cap_tie_mode).strip().lower()

    lines: list[str] = [
        "*HEADING",
        "FastBlade section strip mesh S4R model",
        "*NODE",
    ]

    for nid in sorted(mesh.nodes.keys()):
        x, y, z = mesh.nodes[nid]
        lines.append(f"{nid}, {x:.9e}, {y:.9e}, {z:.9e}")

    lines.append("*ELEMENT, TYPE=S4R, ELSET=EALL")
    for elem in mesh.elements:
        conn = ", ".join(str(n) for n in elem.node_ids)
        lines.append(f"{elem.eid}, {conn}")

    # Panel elsets first (materials / shell sections); skip synthetic CapSkin* sets here.
    for elset_name, eids in mesh.elsets.items():
        if str(elset_name).startswith("CapSkinMasterEls_"):
            continue
        lines.append(f"*ELSET, ELSET={elset_name}")
        lines.extend(_chunk_ids(eids))

    elset_to_panel: dict[str, int] = {}
    for e in mesh.elements:
        elset_name = _sanitize_elset_name(mesh.panel_maps[e.panel_index].panel_label)
        elset_to_panel.setdefault(elset_name, e.panel_index)

    written_mats: set[str] = set()
    for elset_name, pi in sorted(elset_to_panel.items()):
        p = zones[pi]
        mat_name = f"MAT_{elset_name}"
        if mat_name not in written_mats:
            lines.append(f"*MATERIAL, NAME={mat_name}")
            lines.append("*ELASTIC")
            lines.append(f"{float(p.lam.E):.9e}, {float(p.lam.nu):.9e}")
            written_mats.add(mat_name)
        lines.append(f"*SHELL SECTION, ELSET={elset_name}, MATERIAL={mat_name}")
        lines.append(f"{float(p.lam.t):.9e}")

    _emit_cap_skin_ties(lines, mesh, cap_tie_export=cap_tie_export)

    boundary, _anchors = _support_node_ccx_ids(mesh, bc_mode=bc_mode)
    if boundary:
        lines.append("*BOUNDARY")
        for nid, d0, d1 in boundary:
            lines.append(f"{nid}, {d0}, {d1}")

    # CalculiX requires *CLOAD inside the *STEP. Shells are expanded to 3D internally and have
    # no section-force output, so request 3D stress (S) via *EL PRINT and recover resultants by
    # integrating (membrane = mid-surface S * t; see io/calculix_results.py).
    cloads = _cload_entries(mesh, zones, nx_panels, nxy_panels)
    lines.extend(["*STEP", "*STATIC"])
    if cloads:
        lines.append("*CLOAD")
        for nid, dof, val in cloads:
            lines.append(f"{nid}, {dof}, {val:.9e}")
    lines.extend(["*EL PRINT, ELSET=EALL", "S", "*END STEP"])

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def periodic_ref_node_ids(mesh: CcxSectionMesh) -> list[int]:
    """The 6 macro reference-node ids appended after the mesh nodes (modes 0..5).

    Shared by :func:`write_section_inp_periodic` and the periodic recovery so both agree on
    which ccx node ids carry the beam unit-mode amplitudes / reactions.
    """
    base = (max(mesh.nodes.keys()) + 1) if mesh.nodes else 1
    return [base + m for m in range(6)]


def _node_tangents_yz(yz_nodes: np.ndarray) -> np.ndarray:
    """Unit (t_y, t_z) tangent per contour node via central differences along the midline."""
    yz = np.asarray(yz_nodes, dtype=float)
    n = yz.shape[0]
    tan = np.zeros((n, 2))
    if n == 1:
        tan[0] = (1.0, 0.0)
        return tan
    for i in range(n):
        if i == 0:
            d = yz[1] - yz[0]
        elif i == n - 1:
            d = yz[-1] - yz[-2]
        else:
            d = yz[i + 1] - yz[i - 1]
        nrm = float(np.hypot(d[0], d[1]))
        tan[i] = (d / nrm) if nrm > 1e-30 else (1.0, 0.0)
    return tan


def write_section_inp_periodic(
    mesh: CcxSectionMesh,
    zones: list[Any],
    path: str | Path,
    *,
    y_e: float | None = None,
    z_e: float | None = None,
    job_name: str | None = None,
    emit_frd: bool = True,
) -> Path:
    """Write a **periodic** unit-cell S4R deck (6 beam-mode steps) — EXPORT ONLY.

    This is the CalculiX-RVE analogue of the native periodic homogeniser
    (:func:`...engine.homogenisation.compute_k6_periodic_bc`): each inboard (``x=0``) node
    is coupled to its outboard (``x=l_x``) twin by a periodic ``*EQUATION`` whose macro
    displacement jump is carried by **6 reference nodes** (the beam unit modes
    ``ε₀, κ_y, κ_z, T, V_z, V_y``). Six ``*STEP``s drive one reference DOF at a time. Only a
    single node is pinned to remove rigid-body modes — the cut faces are **not** clamped.

    .. note::
       Export only. ``ccx`` is not exercised here; the per-mode section-force recovery and
       K6 assembly (mapping back to ``ShellPanelResultants``) are not implemented yet — see
       ``docs/PERIODIC_CCX_HOMOGENIZATION.md``. The macro-jump coefficients mirror
       ``_macro_jump`` (CCX/B-aligned Cartesian: tangential modes resolve onto ``u_y/u_z``).
    """
    from ..validation.section_centroid import compute_elastic_centroid_from_panels

    out = Path(path)
    job = job_name or out.stem
    l_x = float(mesh.l_x)
    if y_e is None or z_e is None:
        y_e, z_e = compute_elastic_centroid_from_panels(zones)
    y_e = float(y_e)
    z_e = float(z_e)

    lines: list[str] = ["*HEADING", "FastBlade periodic unit-cell S4R model", "*NODE"]
    for nid in sorted(mesh.nodes.keys()):
        x, y, z = mesh.nodes[nid]
        lines.append(f"{nid}, {x:.9e}, {y:.9e}, {z:.9e}")

    # 6 reference nodes carrying the macro amplitudes (placed off the mesh; geometry irrelevant).
    ref_nodes = periodic_ref_node_ids(mesh)  # modes: 0=eps0 1=ky 2=kz 3=T 4=Vz 5=Vy
    for m, rn in enumerate(ref_nodes):
        lines.append(f"{rn}, {-1.0 - m:.9e}, 0.000000000e+00, 0.000000000e+00")
    lines.append("*NSET, NSET=REFNODES")
    lines.append(", ".join(str(n) for n in ref_nodes))

    lines.append("*ELEMENT, TYPE=S4R, ELSET=EALL")
    for elem in mesh.elements:
        conn = ", ".join(str(n) for n in elem.node_ids)
        lines.append(f"{elem.eid}, {conn}")

    for elset_name, eids in mesh.elsets.items():
        lines.append(f"*ELSET, ELSET={elset_name}")
        for i in range(0, len(eids), 16):
            lines.append(", ".join(str(e) for e in eids[i : i + 16]))

    elset_to_panel: dict[str, int] = {}
    for e in mesh.elements:
        elset_name = _sanitize_elset_name(mesh.panel_maps[e.panel_index].panel_label)
        elset_to_panel.setdefault(elset_name, e.panel_index)
    written_mats: set[str] = set()
    for elset_name, pi in sorted(elset_to_panel.items()):
        p = zones[pi]
        mat_name = f"MAT_{elset_name}"
        if mat_name not in written_mats:
            lines.append(f"*MATERIAL, NAME={mat_name}")
            lines.append("*ELASTIC")
            lines.append(f"{float(p.lam.E):.9e}, {float(p.lam.nu):.9e}")
            written_mats.add(mat_name)
        lines.append(f"*SHELL SECTION, ELSET={elset_name}, MATERIAL={mat_name}")
        lines.append(f"{float(p.lam.t):.9e}")

    # Periodic *EQUATION: for each cut-face pair (bottom x=0, top x=l_x), couple top - bottom
    # to the reference DOFs. Translations carry the macro jump; rotations are plain periodic.
    # The "top" node is the dependent (eliminated) DOF; a shared junction node is the top of
    # several panels, so dedup by global top node to avoid the same DOF on two MPC dependent
    # sides (a CalculiX error).
    r0, r1, r2, r3, r4, r5 = (rn for rn in ref_nodes)
    _done_tops: set[int] = set()
    for pi, pm in enumerate(mesh.panel_maps):
        n = len(pm.s_nodes)
        if n == 0:
            continue
        yz_nodes = interp_yz_nodes(pm.s_nodes, zones[pi])
        tans = _node_tangents_yz(yz_nodes)
        for i in range(n):
            gb = pm.global_nodes[i]
            gt = pm.global_nodes[i + n]
            nb = mesh.global_to_ccx_node.get(gb)
            nt = mesh.global_to_ccx_node.get(gt)
            if nb is None or nt is None:
                continue
            if gt in _done_tops:
                continue
            _done_tops.add(gt)
            y_i, z_i = float(yz_nodes[i, 0]), float(yz_nodes[i, 1])
            ty, tz = float(tans[i, 0]), float(tans[i, 1])
            # u_x: top - bottom = eps0*L_x - ky*(z-z_e)*L_x + kz*(y-y_e)*L_x
            _emit_equation(lines, [
                (nt, 1, 1.0), (nb, 1, -1.0),
                (r0, 1, -l_x), (r1, 1, (z_i - z_e) * l_x), (r2, 1, -(y_i - y_e) * l_x),
            ])
            # u_y: top - bottom = ty * (T*L_x + Vz*tz*L_x - Vy*ty*L_x)
            _emit_equation(lines, [
                (nt, 2, 1.0), (nb, 2, -1.0),
                (r3, 1, -ty * l_x), (r4, 1, -ty * tz * l_x), (r5, 1, ty * ty * l_x),
            ])
            # u_z: top - bottom = tz * (T*L_x + Vz*tz*L_x - Vy*ty*L_x)
            _emit_equation(lines, [
                (nt, 3, 1.0), (nb, 3, -1.0),
                (r3, 1, -tz * l_x), (r4, 1, -tz * tz * l_x), (r5, 1, tz * ty * l_x),
            ])
            # Rotations: plain periodic (top == bottom); bending/torsion enter via translation jumps.
            for d in (4, 5, 6):
                _emit_equation(lines, [(nt, d, 1.0), (nb, d, -1.0)])

    # Rigid-body removal: do not pin any cut-face mesh node — every physical node lies on
    # x=0 or x=L_x, and fixing translations there conflicts with macro bending/torsion modes.
    # MITC4 removes one reference node from the free set instead of clamping to zero; here the
    # periodic *EQUATION* coupling plus the driven reference nodes provide the datum.
    # Pin unused reference-node DOFs 2-6 (macro amplitude lives on DOF 1 only).
    lines.append("*BOUNDARY")
    for rn in ref_nodes:
        lines.append(f"{rn}, 2, 6, 0.0")

    # Six steps: drive one reference DOF = 1 (others 0); print per-element section forces (SF)
    # and the reference-node reactions (RF). The 6x6 reaction matrix is the section K6 (the
    # conjugate beam forces for each unit macro mode); the per-element SF are the unit-mode
    # fields superposed by the solved mode amplitudes during recovery.
    mode_names = ["eps0", "ky", "kz", "torsion", "Vz", "Vy"]
    for m in range(6):
        lines.append(f"** mode {m}: {mode_names[m]}")
        lines.extend(["*STEP", "*STATIC", "*BOUNDARY, OP=NEW"])
        for mm, rnn in enumerate(ref_nodes):
            lines.append(f"{rnn}, 1, 1, {1.0 if mm == m else 0.0:.1f}")
        step_lines = [
            "*EL PRINT, ELSET=EALL",
            "S",
            "*NODE PRINT, NSET=REFNODES",
            "RF",
        ]
        if emit_frd:
            step_lines = ["*EL FILE", "S", *step_lines]
        lines.extend([*step_lines, "*END STEP"])

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _emit_equation(lines: list[str], terms: list[tuple[int, int, float]]) -> None:
    """Append a CalculiX ``*EQUATION`` (terms sum to zero)."""
    lines.append("*EQUATION")
    lines.append(str(len(terms)))
    lines.append(", ".join(f"{nid}, {dof}, {coeff:.9e}" for nid, dof, coeff in terms))


__all__ = ["write_section_inp", "write_section_inp_periodic", "periodic_ref_node_ids"]
