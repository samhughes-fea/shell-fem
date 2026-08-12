"""CalculiX deck writer and solve driver for the clamped root-segment diagnostic.

Deck layout
-----------
* S4R shell elements on the lofted multi-row mesh.
* Full clamp (DOFs 1–6) on ``ROOT_RING``.
* Outboard beam resultants applied as statically equivalent ``*CLOAD`` on
  ``OUTBOARD_RING`` nodes (CalculiX forbids ``*RIGID BODY`` / nodal
  ``*DISTRIBUTING`` on expanded shell nodes).
* Isotropic-equivalent ``*SHELL SECTION`` (same v1 material export as unit-strip decks).

Standalone / diagnostic only — not registered in production theory dispatch.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .mesh import _sanitize_elset_name
from .results import (
    RootSegmentMappedResultants,
    map_root_segment_stresses_to_resultants,
    parse_ccx_dat_node_reactions,
    parse_ccx_dat_stresses,
)
from .root_segment_mesh import CcxRootSegmentMesh, build_ccx_root_segment_mesh


def _chunk_ids(ids: list[int], width: int = 16) -> list[str]:
    lines: list[str] = []
    for i in range(0, len(ids), width):
        lines.append(", ".join(str(n) for n in ids[i : i + width]))
    return lines


def _outboard_node_weights(
    mesh: CcxRootSegmentMesh,
    zones: list[Any],
) -> list[tuple[int, float, float, float]]:
    """Return ``(ccx_nid, y, z, weight=t*ds_trib)`` for unique outboard contour nodes."""
    entries: list[tuple[int, float, float, float]] = []
    seen: set[int] = set()
    for pi, pm in enumerate(mesh.panel_maps):
        n = len(pm.s_nodes)
        if n < 2 or pi >= len(zones):
            continue
        t = float(zones[pi].lam.t)
        # Outboard layer stashed in global_nodes[n : 2n]
        for li in range(n):
            if li + n >= len(pm.global_nodes):
                continue
            nid = int(pm.global_nodes[li + n])
            if nid < 0 or nid in seen:
                continue
            seen.add(nid)
            _x, y, z = mesh.nodes[nid]
            # Tributary contour length
            if li == 0:
                ds = 0.5 * abs(float(pm.s_nodes[1] - pm.s_nodes[0]))
            elif li == n - 1:
                ds = 0.5 * abs(float(pm.s_nodes[-1] - pm.s_nodes[-2]))
            else:
                ds = 0.5 * abs(float(pm.s_nodes[li + 1] - pm.s_nodes[li - 1]))
            w = max(t * ds, 0.0)
            entries.append((nid, float(y), float(z), w))
    return entries


def outboard_cloads_from_resultants(
    mesh: CcxRootSegmentMesh,
    zones: list[Any],
    *,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Statically equivalent nodal ``*CLOAD`` on the outboard ring [nid, dof, value].

    Translations (``N``, ``Vy``, ``Vz``) are distributed by wall weight ``t·ds``.
    Bending moments ``My``/``Mz`` add an axial force pattern proportional to
    ``z·w`` / ``y·w`` (zero net force). Torsion ``T`` applies tangential forces
    about the weighted centre (``ty, tz) ∝ (-(z-zc), y-yc)``.

    This is an approximate Saint-Venant cut load — only trusted a diameter or more
    inboard of the outboard face (see ``docs/ROOT_SEGMENT_CCX_DIAGNOSTIC.md``).
    """
    nodes = _outboard_node_weights(mesh, zones)
    if not nodes:
        return []
    w = np.array([e[3] for e in nodes], dtype=float)
    y = np.array([e[1] for e in nodes], dtype=float)
    z = np.array([e[2] for e in nodes], dtype=float)
    nids = [e[0] for e in nodes]
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        w = np.ones(len(nodes), dtype=float)
        w_sum = float(len(nodes))
    wn = w / w_sum
    yc = float(np.sum(wn * y))
    zc = float(np.sum(wn * z))
    y0 = y - yc
    z0 = z - zc

    fx = float(N) * wn
    fy = float(Vy) * wn
    fz = float(Vz) * wn

    # My about y at centre → couple Fx ∝ z; scale so sum(Fx * z0) = My
    iz = float(np.sum(w * z0 * z0))
    if abs(float(My)) > 0.0 and iz > 1e-30:
        fx = fx + (float(My) / iz) * (w * z0)
    # Mz about z → couple Fx ∝ -y; sum(Fx * (-y0)) = Mz  ⇒  sum(Fx * y0) = -Mz
    iy = float(np.sum(w * y0 * y0))
    if abs(float(Mz)) > 0.0 and iy > 1e-30:
        fx = fx - (float(Mz) / iy) * (w * y0)

    # Torsion: tangential force about centre; T = sum(y0*Fz - z0*Fy) with F∝(-z0, y0)
    # Let F_y = c * (-z0)*w, F_z = c * y0 * w → T_contrib = c * sum(w*(y0^2+z0^2))
    if abs(float(T)) > 0.0:
        r2 = y0 * y0 + z0 * z0
        denom = float(np.sum(w * r2))
        if denom > 1e-30:
            c = float(T) / denom
            fy = fy + c * (-z0) * w
            fz = fz + c * y0 * w

    loads: dict[tuple[int, int], float] = {}
    for i, nid in enumerate(nids):
        for dof, val in ((1, float(fx[i])), (2, float(fy[i])), (3, float(fz[i]))):
            if abs(val) <= 0.0:
                continue
            key = (nid, dof)
            loads[key] = loads.get(key, 0.0) + val
    return [(nid, dof, val) for (nid, dof), val in sorted(loads.items())]


def write_root_segment_inp(
    mesh: CcxRootSegmentMesh,
    zones: list[Any],
    path: str | Path,
    *,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    job_name: str | None = None,
    emit_frd: bool = False,
) -> Path:
    """Write a clamped root-segment S4R CalculiX deck with outboard nodal resultants."""
    out = Path(path)
    job = job_name or out.stem
    lines: list[str] = [
        "*HEADING",
        f"FastBlade root-segment diagnostic S4R model ({job})",
        "*NODE",
    ]
    # Skip unused REF/ROT helper nodes — they are not referenced by BCs/loads.
    skip_ids = {mesh.ref_node_id, mesh.rot_node_id}
    for nid in sorted(mesh.nodes.keys()):
        if nid in skip_ids:
            continue
        x, y, z = mesh.nodes[nid]
        lines.append(f"{nid}, {x:.9e}, {y:.9e}, {z:.9e}")

    lines.append("*ELEMENT, TYPE=S4R, ELSET=EALL")
    for elem in mesh.elements:
        conn = ", ".join(str(n) for n in elem.node_ids)
        lines.append(f"{elem.eid}, {conn}")

    for elset_name, eids in mesh.elsets.items():
        lines.append(f"*ELSET, ELSET={elset_name}")
        lines.extend(_chunk_ids(eids))

    for nset_name in ("ROOT_RING", "OUTBOARD_RING"):
        ids = mesh.nsets.get(nset_name, [])
        if not ids:
            continue
        lines.append(f"*NSET, NSET={nset_name}")
        lines.extend(_chunk_ids(ids))

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

    lines.append("*BOUNDARY")
    lines.append("ROOT_RING, 1, 6")

    cloads = outboard_cloads_from_resultants(
        mesh, zones, N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T
    )

    lines.extend(["*STEP", "*STATIC"])
    if cloads:
        lines.append("*CLOAD")
        for nid, dof, val in cloads:
            lines.append(f"{nid}, {dof}, {val:.9e}")
    if emit_frd:
        lines.extend(["*EL FILE", "S"])
    lines.extend(
        [
            "*EL PRINT, ELSET=EALL",
            "S",
            "*NODE PRINT, NSET=ROOT_RING",
            "RF",
            "*NODE PRINT, NSET=OUTBOARD_RING",
            "RF",
            "*END STEP",
        ]
    )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _require_ccx(ccx_exe: str = "ccx") -> str:
    path = shutil.which(ccx_exe)
    if path is None:
        raise RuntimeError(
            f"CalculiX executable {ccx_exe!r} not found on PATH; "
            "required for root-segment diagnostic"
        )
    return path


def run_root_segment_ccx(
    zones: list[Any],
    *,
    work_dir: str | Path,
    segment_length_m: float | None = None,
    n_span_rows: int = 8,
    diameter_multiples: float = 2.5,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    section_strip_mesh: Any | None = None,
    n_elements_per_panel: int | Mapping[int, int] | None = None,
    target_element_length_m: float | None = None,
    station_panels: list[list[Any]] | None = None,
    station_span_coords: np.ndarray | None = None,
    ccx_exe: str = "ccx",
    job_name: str = "root_segment",
    emit_frd: bool = False,
) -> tuple[CcxRootSegmentMesh, RootSegmentMappedResultants, dict[str, Any], Path]:
    """Build mesh, write deck, run ``ccx``, map root-adjacent resultants + decay profile.

    Returns ``(mesh, mapped, diagnostics, inp_path)``.
    """
    from blade_structure_utilities.mesh_utilities.section_mesh import (
        build_section_strip_mesh_from_panels,
    )
    from .root_segment_mesh import estimate_root_diameter_m

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    ccx_path = _require_ccx(ccx_exe)

    diam = estimate_root_diameter_m(zones)
    L = float(segment_length_m) if segment_length_m is not None else float(diameter_multiples) * diam
    if section_strip_mesh is None:
        section_strip_mesh = build_section_strip_mesh_from_panels(
            zones,
            n_elements_per_panel=n_elements_per_panel,
            target_element_length_m=target_element_length_m,
        )
    mesh = build_ccx_root_segment_mesh(
        zones,
        section_strip_mesh=section_strip_mesh,
        segment_length_m=L,
        n_span_rows=n_span_rows,
        station_panels=station_panels,
        station_span_coords=station_span_coords,
    )
    inp_path = work / f"{job_name}.inp"
    write_root_segment_inp(
        mesh,
        zones,
        inp_path,
        N=N,
        Vy=Vy,
        Vz=Vz,
        My=My,
        Mz=Mz,
        T=T,
        job_name=job_name,
        emit_frd=emit_frd,
    )
    completed = subprocess.run(
        [ccx_path, job_name],
        cwd=str(work),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        tail = (completed.stdout or "")[-2000:]
        raise RuntimeError(
            f"ccx {job_name!r} failed with exit {completed.returncode} in {work}\n{tail}"
        )
    dat_path = work / f"{job_name}.dat"
    stress_rows = parse_ccx_dat_stresses(dat_path) if dat_path.is_file() else []
    mapped = map_root_segment_stresses_to_resultants(mesh, zones, stress_rows)
    root_rf = (
        parse_ccx_dat_node_reactions(dat_path, nset_hint="ROOT_RING")
        if dat_path.is_file()
        else {}
    )

    cloads = outboard_cloads_from_resultants(
        mesh, zones, N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T
    )
    sum_fx_app = sum(v for _n, d, v in cloads if d == 1)
    sum_fy_app = sum(v for _n, d, v in cloads if d == 2)
    sum_fz_app = sum(v for _n, d, v in cloads if d == 3)
    sum_rx = sum(v[0] for v in root_rf.values())
    sum_ry = sum(v[1] for v in root_rf.values())
    sum_rz = sum(v[2] for v in root_rf.values())
    diag: dict[str, Any] = {
        "inp_path": str(inp_path.resolve()),
        "dat_path": str(dat_path.resolve()) if dat_path.is_file() else None,
        "n_stress_rows": len(stress_rows),
        "segment_length_m": mesh.segment_length_m,
        "root_diameter_m": mesh.root_diameter_m,
        "n_span_rows": mesh.n_span_rows,
        "applied_cload_sum": (sum_fx_app, sum_fy_app, sum_fz_app),
        "applied_resultants": {
            "N": float(N),
            "Vy": float(Vy),
            "Vz": float(Vz),
            "My": float(My),
            "Mz": float(Mz),
            "T": float(T),
        },
        "root_rf_sum": (sum_rx, sum_ry, sum_rz),
        "n_root_rf_nodes": len(root_rf),
        "n_cload_entries": len(cloads),
        "decay_profile": mapped.decay_profile,
    }
    return mesh, mapped, diag, inp_path


__all__ = [
    "outboard_cloads_from_resultants",
    "run_root_segment_ccx",
    "write_root_segment_inp",
]
