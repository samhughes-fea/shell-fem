"""Parse CalculiX ccx output into shell panel resultants."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import FieldProvenance, ProvenanceKind, ShellPanelResultants
from .mesh import CcxSectionMesh


@dataclass(frozen=True)
class CcxSectionForceRow:
    eid: int
    n11: float
    n22: float
    n12: float
    m11: float
    m22: float
    m12: float
    q1: float
    q2: float


def parse_ccx_dat_section_forces(dat_path: str | Path) -> list[CcxSectionForceRow]:
    """Parse section force rows from a CalculiX ``.dat`` file (best-effort)."""
    text = Path(dat_path).read_text(encoding="utf-8", errors="replace")
    rows: list[CcxSectionForceRow] = []

    # CalculiX dat: element lines often " elem  integ.pt.  NXX  NYY  NXY  MXX  ..."
    block_re = re.compile(
        r"section\s+force|internal\s+force",
        re.IGNORECASE,
    )
    number_line = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+"
        r"([-+eE\d.]+)\s+([-+eE\d.]+)\s+([-+eE\d.]+)\s+"
        r"([-+eE\d.]+)\s+([-+eE\d.]+)\s+([-+eE\d.]+)\s+"
        r"([-+eE\d.]+)\s+([-+eE\d.]+)",
        re.MULTILINE,
    )

    in_block = False
    for line in text.splitlines():
        if block_re.search(line):
            in_block = True
            continue
        if in_block and line.strip() == "":
            in_block = False
        if not in_block:
            continue
        m = number_line.match(line)
        if not m:
            continue
        eid = int(m.group(1))
        vals = [float(m.group(i)) for i in range(3, 11)]
        rows.append(
            CcxSectionForceRow(
                eid=eid,
                n11=vals[0],
                n22=vals[1],
                n12=vals[2],
                m11=vals[3],
                m22=vals[4],
                m12=vals[5],
                q1=vals[6],
                q2=vals[7],
            )
        )

    if rows:
        return rows

    # Fallback: any line starting with element id + 8 floats
    simple = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+"
        r"([-+eE\d.]+)\s+([-+eE\d.]+)\s+([-+eE\d.]+)\s+"
        r"([-+eE\d.]+)\s+([-+eE\d.]+)\s+([-+eE\d.]+)\s+"
        r"([-+eE\d.]+)\s+([-+eE\d.]+)\s*$"
    )
    for line in text.splitlines():
        m = simple.match(line)
        if not m:
            continue
        eid = int(m.group(1))
        vals = [float(m.group(i)) for i in range(3, 11)]
        rows.append(
            CcxSectionForceRow(
                eid=eid,
                n11=vals[0],
                n22=vals[1],
                n12=vals[2],
                m11=vals[3],
                m22=vals[4],
                m12=vals[5],
                q1=vals[6],
                q2=vals[7],
            )
        )
    return rows


@dataclass(frozen=True)
class CcxElementStress:
    """Mean (mid-surface) 3D stress for one expanded-shell element, in global axes."""

    eid: int
    sxx: float
    syy: float
    szz: float
    sxy: float
    sxz: float
    syz: float


_STRESS_HEADER = re.compile(r"stresses\s*\(elem", re.IGNORECASE)
_STRESS_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+" + r"\s+".join([r"([-+eE\d.]+)"] * 6)
)


def _parse_stress_block(lines: list[str]) -> list[CcxElementStress]:
    """Average integration-point stresses per element id within one *EL PRINT S block."""
    acc: dict[int, list[np.ndarray]] = {}
    for line in lines:
        m = _STRESS_ROW.match(line)
        if not m:
            continue
        eid = int(m.group(1))
        acc.setdefault(eid, []).append(np.array([float(m.group(i)) for i in range(3, 9)]))
    out: list[CcxElementStress] = []
    for eid, pts in acc.items():
        s = np.mean(np.stack(pts, axis=0), axis=0)
        out.append(CcxElementStress(eid, *s.tolist()))
    return out


def parse_ccx_dat_stresses(dat_path: str | Path) -> list[CcxElementStress]:
    """Parse element 3D stresses (``*EL PRINT, S``) from a single-step ``.dat`` file.

    CalculiX has no shell section-force output; shells are expanded to 3D and report ``S``
    (sxx,syy,szz,sxy,sxz,syz) at integration points. This averages the points per element
    (mid-surface), from which membrane/transverse-shear resultants are recovered downstream.
    """
    text = Path(dat_path).read_text(encoding="utf-8", errors="replace")
    block: list[str] = []
    collecting = False
    for line in text.splitlines():
        if _STRESS_HEADER.search(line):
            collecting = True
            continue
        if collecting:
            block.append(line)
    return _parse_stress_block(block)


def map_ccx_stresses_to_panel_resultants(
    mesh: CcxSectionMesh,
    zones: list[Any],
    stress_rows: list[CcxElementStress],
) -> tuple[list[list[ShellPanelResultants]], list[dict]]:
    """Map ccx element 3D stresses → per-panel ``ShellPanelResultants`` (membrane + transverse shear).

    Stresses are in global axes; the strip span is global x, the contour lies in the y–z plane.
    Per element with unit contour tangent ``(ty, tz)`` and wall normal ``(ny, nz) = (-tz, ty)``:

    * ``N11``  = sxx · t                              (spanwise/axial membrane)
    * ``N12`` = (sxy·ty + sxz·tz) · t                (shear flow: x-face shear along the contour)
    * ``N22``  = (syy·ty² + 2·syz·ty·tz + szz·tz²) · t (contour-direction membrane)
    * ``Q1``  = (sxy·ny + sxz·nz) · t                (transverse shear τ_xζ · t)
    * ``Q2``  = ((syy·ny + syz·nz)·ty + (syz·ny + szz·nz)·tz) · t  (transverse shear τ_sζ · t)

    Per-wall bending ``M11/M22/M12`` need through-thickness stress (mid-surface bending stress is
    ~0). They are initially tagged ``RESERVED`` (``needs .frd through-thickness``); when
    ``emit_frd=True``, ``apply_frd_bending_to_panel_resultants`` upgrades available eids to
    ``CALCULIX_CCX`` or records an explicit skip reason on the RESERVED note.
    """
    by_eid = {r.eid: r for r in stress_rows}
    prov_kind = ProvenanceKind.CALCULIX_CCX
    keys = ("N11", "N22", "N12", "Q1", "Q2")
    all_results: list[list[ShellPanelResultants]] = []
    all_diag: list[dict] = []
    for pi, pm in enumerate(mesh.panel_maps):
        if len(pm.s_nodes) == 0:
            all_results.append([]); all_diag.append({}); continue
        thickness = float(zones[pi].lam.t)
        panel_elems = sorted(
            (e for e in mesh.elements if e.panel_index == pi),
            key=lambda e: e.local_element_index,
        )
        results: list[ShellPanelResultants] = []
        for elem in panel_elems:
            r = by_eid.get(elem.eid)
            n0, n1 = elem.node_ids[0], elem.node_ids[1]
            (_x0, y0, z0) = mesh.nodes[n0]
            (_x1, y1, z1) = mesh.nodes[n1]
            dy, dz = y1 - y0, z1 - z0
            length = float(np.hypot(dy, dz))
            if length <= 0.0:
                raise ValueError(
                    f"CCX element {elem.eid} on panel {pi}: zero-length edge "
                    f"(nodes {n0}/{n1}); refusing silent length=1.0 (FEA-CCX-004)"
                )
            ty, tz = dy / length, dz / length
            ny, nz = -tz, ty
            if r is None:
                results.append(ShellPanelResultants(
                    N11=0.0, N22=0.0, N12=0.0, M11=0.0, M22=0.0, M12=0.0, Q1=0.0, Q2=0.0,
                    provenance={k: FieldProvenance(ProvenanceKind.PLACEHOLDER, "ccx S missing")
                                for k in ("N11", "N22", "N12", "M11", "M22", "M12", "Q1", "Q2")},
                    thickness_m=thickness, panel_label=pm.panel_label, panel_index=pi,
                    station_index=elem.local_element_index,
                ))
                continue
            nx = r.sxx * thickness
            nxy = (r.sxy * ty + r.sxz * tz) * thickness
            ny_res = (r.syy * ty * ty + 2.0 * r.syz * ty * tz + r.szz * tz * tz) * thickness
            qx = (r.sxy * ny + r.sxz * nz) * thickness
            # Traction on mid-surface (normal n) dotted with contour tangent → τ_sζ.
            qy = ((r.syy * ny + r.syz * nz) * ty + (r.syz * ny + r.szz * nz) * tz) * thickness
            prov = {k: FieldProvenance(prov_kind, f"ccx S->{k}") for k in keys}
            prov.update({k: FieldProvenance(ProvenanceKind.RESERVED, "needs .frd through-thickness")
                         for k in ("M11", "M22", "M12")})
            results.append(ShellPanelResultants(
                N11=nx, N22=ny_res, N12=nxy, M11=0.0, M22=0.0, M12=0.0, Q1=qx, Q2=qy,
                provenance=prov,
                sigma_11_pa=r.sxx, tau_12_pa=nxy, q_n_per_m=nxy, thickness_m=thickness,
                panel_label=pm.panel_label, panel_index=pi, station_index=elem.local_element_index,
            ))
        all_results.append(results)
        all_diag.append({"solver": "global_coupled_ccx", "element_kernel": "calculix_ccx_stress",
                         "n_parsed_stresses": sum(1 for e in mesh.elements if e.panel_index == pi and e.eid in by_eid)})
    return all_results, all_diag


@dataclass(frozen=True)
class CcxFrdIpStress:
    """Through-thickness integration-point stress from a CalculiX ``.frd`` shell record."""

    eid: int
    ip: int
    sxx: float
    syy: float
    szz: float
    sxy: float
    syz: float
    sxz: float


_FRD_STRESS_MARKER = re.compile(r"^\s*-4\s+STRESS", re.IGNORECASE)
# ccx 2.23 packed ``-1`` rows: ``i10`` internal element id + ``ip`` digit fused with first ``e12.5``.
_FRD_FLOAT_PACKED = re.compile(r"[-+]?\d*\.\d+E[-+]\d{2,3}", re.IGNORECASE)


def _parse_frd_floats_packed(tail: str) -> list[float]:
    """Scan up to six stress components from the packed tail (handles ``E+010`` overlap)."""
    floats: list[float] = []
    pos = 0
    while len(floats) < 6 and pos < len(tail):
        m = _FRD_FLOAT_PACKED.search(tail, pos)
        if not m:
            break
        floats.append(float(m.group()))
        pos = m.end()
    return floats


def _frd_internal_ip_from_record(rec: str) -> tuple[int, int]:
    """Decode ccx expanded-shell ``(internal_eid, ip)`` from a packed ``-1`` record body."""
    internal_eid = int(rec[0:10])
    if len(rec) > 11 and rec[10].isdigit() and rec[11] == ".":
        return internal_eid, int(rec[10])
    return internal_eid, 0


def _frd_internal_ip_from_node_key(node_key: int) -> tuple[int, int]:
    """Decode spaced-format node label ``10 * internal + ip``."""
    if node_key >= 100:
        return divmod(node_key, 10)
    return int(node_key), 0


def _parse_frd_minus1_line(line: str) -> tuple[int, int, list[float]] | None:
    """Parse a CalculiX FRD ``-1`` stress row → ``(internal_eid, ip, six components)``."""
    s = line.lstrip()
    if not s.startswith("-1"):
        return None
    rec = s[2:]
    parts = rec.split()
    if len(parts) >= 7:
        node_key = int(parts[0])
        internal_eid, ip = _frd_internal_ip_from_node_key(node_key)
        return internal_eid, ip, [float(x) for x in parts[1:7]]
    if len(rec) < 10 or not rec[:10].strip():
        return None
    internal_eid, ip = _frd_internal_ip_from_record(rec)
    floats = _parse_frd_floats_packed(rec[10:])
    if not floats:
        return None
    while len(floats) < 6:
        floats.append(0.0)
    return internal_eid, ip, floats[:6]


def ccx_frd_element_base(mesh: Any) -> int:
    """First ccx expanded-shell element id (= max mesh node id + periodic ref nodes)."""
    from .section_export import periodic_ref_node_ids

    return int(max(mesh.nodes)) + len(periodic_ref_node_ids(mesh))


@dataclass(frozen=True)
class FrdCcxNodeIndex:
    """Node membership parsed from the ccx ``3C`` element-connectivity block in a ``.frd`` file."""

    user_nodes: dict[int, frozenset[int]]
    node_to_users: dict[int, tuple[int, ...]]


_FRD_3C_MARKER = re.compile(r"^\s*3C\s", re.IGNORECASE)


def parse_ccx_frd_3c_node_index(frd_path: str | Path) -> FrdCcxNodeIndex | None:
    """Parse ccx ``3C`` connectivity → user element id ↔ expanded node ids."""
    text = Path(frd_path).read_text(encoding="utf-8", errors="replace")
    user_nodes: dict[int, set[int]] = {}
    in_block = False
    cur_eid: int | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if _FRD_3C_MARKER.match(line) and not stripped.startswith("-4"):
            in_block = True
            continue
        if in_block and stripped == "-3":
            break
        if not in_block:
            continue
        if line.lstrip().startswith("-1"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                cur_eid = int(parts[1])
                user_nodes.setdefault(cur_eid, set())
            continue
        if line.lstrip().startswith("-2") and cur_eid is not None:
            for tok in line.split()[1:]:
                if tok.lstrip("-").isdigit():
                    user_nodes[cur_eid].add(int(tok))

    if not user_nodes:
        return None
    frozen = {eid: frozenset(nodes) for eid, nodes in user_nodes.items()}
    node_to_users: dict[int, list[int]] = {}
    for eid, nodes in frozen.items():
        for nid in nodes:
            node_to_users.setdefault(nid, []).append(eid)
    return FrdCcxNodeIndex(
        user_nodes=frozen,
        node_to_users={nid: tuple(sorted(eids)) for nid, eids in node_to_users.items()},
    )


def parse_ccx_frd_element_map(
    frd_path: str | Path,
    *,
    element_base: int,
    n_mesh_elements: int,
) -> dict[int, int]:
    """Build expanded FRD entity id → user mesh ``eid`` using ``3C`` node membership."""
    index = parse_ccx_frd_3c_node_index(frd_path)
    if index is None:
        return {}
    entity_ids: set[int] = set()
    for nodes in index.user_nodes.values():
        entity_ids.update(nodes)
    return {
        eid: mesh_eid
        for eid in entity_ids
        if (mesh_eid := _frd_to_mesh_eid(
            eid,
            element_base=element_base,
            n_mesh_elements=n_mesh_elements,
            frd_index=index,
        ))
        is not None
    }


def _frd_to_mesh_eid(
    internal_eid: int,
    *,
    element_base: int,
    n_mesh_elements: int,
    frd_index: FrdCcxNodeIndex | None = None,
) -> int | None:
    """Map ccx FRD stress entity id to user mesh ``eid``."""
    internal = int(internal_eid)
    n_max = int(n_mesh_elements)
    base = int(element_base)

    me = internal - base
    if 1 <= me <= n_max:
        return me

    if frd_index is not None:
        for user_eid, nodes in frd_index.user_nodes.items():
            if internal in nodes and 1 <= int(user_eid) <= n_max:
                return int(user_eid)
        cands = frd_index.node_to_users.get(internal)
        if cands:
            if len(cands) == 1:
                return int(cands[0])
            for cand in cands:
                if internal - base == cand:
                    return int(cand)
            return int(
                min(
                    cands,
                    key=lambda c: (
                        0 if internal == min(frd_index.user_nodes[c]) else 1,
                        c,
                    ),
                )
            )

    if internal >= 10:
        me_alt, ip_rem = divmod(internal, 10)
        if 1 <= me_alt <= n_max and 1 <= ip_rem <= 8:
            return me_alt
        if 1 <= me_alt <= n_max:
            return me_alt
    return None


def list_unmapped_frd_entities(
    frd_by_eid: dict[int, list],
    *,
    element_base: int,
    n_mesh_elements: int,
    frd_index: FrdCcxNodeIndex | None = None,
) -> list[int]:
    """Diagnostic: FRD entity ids that fail mesh remap."""
    unmapped: list[int] = []
    for internal_eid in frd_by_eid:
        if _frd_to_mesh_eid(
            internal_eid,
            element_base=element_base,
            n_mesh_elements=n_mesh_elements,
            frd_index=frd_index,
        ) is None:
            unmapped.append(int(internal_eid))
    return unmapped


def _remap_frd_eids_to_mesh(
    frd_by_eid: dict[int, list[CcxFrdIpStress]],
    *,
    element_base: int,
    n_mesh_elements: int | None = None,
    frd_index: FrdCcxNodeIndex | None = None,
) -> dict[int, list[CcxFrdIpStress]]:
    """Map ccx expanded-shell internal ids to user mesh ``eid``."""
    n_max = int(n_mesh_elements) if n_mesh_elements is not None else 10**9
    out: dict[int, list[CcxFrdIpStress]] = {}
    for internal_eid, rows in frd_by_eid.items():
        mesh_eid = _frd_to_mesh_eid(
            internal_eid,
            element_base=element_base,
            n_mesh_elements=n_max,
            frd_index=frd_index,
        )
        if mesh_eid is None:
            continue
        out.setdefault(mesh_eid, []).extend(rows)
    return out


def parse_ccx_frd_stress_blocks(frd_path: str | Path) -> list[list[CcxFrdIpStress]]:
    """Parse per-step STRESS blocks from a CalculiX ``.frd`` file.

    Shell expanded elements encode ``node_key = 10 * element_id + integration_point``.
    """
    text = Path(frd_path).read_text(encoding="utf-8", errors="replace")
    blocks: list[list[CcxFrdIpStress]] = []
    in_block = False
    cur: list[CcxFrdIpStress] = []

    def _flush() -> None:
        nonlocal cur
        if cur:
            blocks.append(cur)
        cur = []

    for line in text.splitlines():
        if _FRD_STRESS_MARKER.match(line):
            _flush()
            in_block = True
            continue
        if in_block and line.strip().startswith("-4"):
            _flush()
            in_block = False
            continue
        if not in_block:
            continue
        parsed = _parse_frd_minus1_line(line)
        if parsed is None:
            continue
        internal_eid, ip, vals = parsed
        if internal_eid <= 0:
            continue
        cur.append(CcxFrdIpStress(internal_eid, ip, *vals))

    _flush()
    return blocks


def superpose_frd_stress_blocks(
    blocks: list[list[CcxFrdIpStress]], q_beam: np.ndarray
) -> dict[int, list[CcxFrdIpStress]]:
    """Superpose per-mode FRD IP stresses by solved beam-mode amplitudes ``q_beam``."""
    q = np.asarray(q_beam, dtype=float)
    acc: dict[tuple[int, int], np.ndarray] = {}
    for step_idx, rows in enumerate(blocks):
        if step_idx >= len(q):
            break
        w = float(q[step_idx])
        for r in rows:
            key = (r.eid, r.ip)
            acc.setdefault(key, np.zeros(6))
            acc[key] += w * np.array([r.sxx, r.syy, r.szz, r.sxy, r.syz, r.sxz])
    by_eid: dict[int, list[CcxFrdIpStress]] = {}
    for (eid, ip), vals in sorted(acc.items()):
        by_eid.setdefault(eid, []).append(CcxFrdIpStress(eid, ip, *vals.tolist()))
    return by_eid


def bending_moments_from_frd_ips(
    ip_rows: list[CcxFrdIpStress],
    *,
    thickness: float,
    ty: float,
    tz: float,
) -> tuple[float, float, float]:
    """Estimate ``(M11, M22, M12)`` [N·m/m] from through-thickness IP stress spread.

    * ``M11`` — spanwise ``sxx`` gradient (bending about contour axis).
    * ``M22`` — contour-direction normal stress gradient (same projection as membrane ``N22``).
    * ``M12`` — in-plane shear spread projected onto the strip tangent.
    """
    if not ip_rows or thickness <= 0.0:
        return 0.0, 0.0, 0.0
    t = float(thickness)
    t2 = t * t / 12.0
    ty2, tz2 = ty * ty, tz * tz
    sxx_vals = [r.sxx for r in ip_rows]
    sxy_vals = [r.sxy for r in ip_rows]
    sxz_vals = [r.sxz for r in ip_rows]
    sig_cont = [
        r.syy * ty2 + 2.0 * r.syz * ty * tz + r.szz * tz2 for r in ip_rows
    ]
    mx = (max(sxx_vals) - min(sxx_vals)) * t2
    my = (max(sig_cont) - min(sig_cont)) * t2
    dsxy = (max(sxy_vals) - min(sxy_vals)) * ty + (max(sxz_vals) - min(sxz_vals)) * tz
    mxy = dsxy * t2
    return mx, my, mxy


def _frd_stress_magnitude_pa(ip_rows: list[CcxFrdIpStress]) -> float:
    """Peak absolute 3D stress component across IP rows [Pa]."""
    peak = 0.0
    for r in ip_rows:
        for val in (r.sxx, r.syy, r.szz, r.sxy, r.sxz, r.syz):
            if np.isfinite(val):
                peak = max(peak, abs(float(val)))
    return peak


def _frd_bending_bounded(
    mx: float,
    my: float,
    mxy: float,
    ip_rows: list[CcxFrdIpStress],
    *,
    thickness: float,
) -> tuple[float, float, float] | None:
    """Return bending triplet if finite and within a physical ``Δσ·t²/12`` bound."""
    t = float(thickness)
    if t <= 0.0 or not ip_rows:
        return None
    vals = (mx, my, mxy)
    if not all(np.isfinite(v) for v in vals):
        return None
    peak_s = _frd_stress_magnitude_pa(ip_rows)
    if peak_s <= 0.0 or peak_s > 1.0e15:
        return None
    bound = peak_s * t * t / 12.0
    if bound <= 0.0:
        return None
    if max(abs(mx), abs(my), abs(mxy)) > 100.0 * bound:
        return None
    return mx, my, mxy


def apply_frd_bending_to_panel_resultants(
    results: list[list[ShellPanelResultants]],
    mesh: CcxSectionMesh,
    frd_by_eid: dict[int, list[CcxFrdIpStress]],
) -> dict[str, int]:
    """In-place: populate ``M11/M22/M12`` from superposed FRD IP stresses where available.

    Returns a small coverage diagnostic::

        {
          "n_elements": ...,
          "n_applied": ...,          # M11/M22/M12 upgraded to CALCULIX_CCX
          "n_missing_ip": ...,      # no FRD rows for mesh eid after remap
          "n_rejected_bound": ...,  # IP present but failed physical bound
        }

    Elements that remain without FRD bending keep ``RESERVED`` provenance with an
    explicit skip reason (missing IP vs bound reject) so stress-only callers that
    requested ``.frd`` do not silently leave a bare ``needs .frd`` placeholder.
    """
    prov_kind = ProvenanceKind.CALCULIX_CCX
    stats = {
        "n_elements": 0,
        "n_applied": 0,
        "n_missing_ip": 0,
        "n_rejected_bound": 0,
    }
    for pi, pm in enumerate(mesh.panel_maps):
        if pi >= len(results) or not results[pi]:
            continue
        panel_elems = sorted(
            (e for e in mesh.elements if e.panel_index == pi),
            key=lambda e: e.local_element_index,
        )
        for elem, res in zip(panel_elems, results[pi]):
            stats["n_elements"] += 1
            ip_rows = frd_by_eid.get(elem.eid)
            if not ip_rows:
                stats["n_missing_ip"] += 1
                for key in ("M11", "M22", "M12"):
                    res.provenance[key] = FieldProvenance(
                        ProvenanceKind.RESERVED, "frd requested but no IP for eid",
                    )
                continue
            n0, n1 = elem.node_ids[0], elem.node_ids[1]
            (_x0, y0, z0) = mesh.nodes[n0]
            (_x1, y1, z1) = mesh.nodes[n1]
            dy, dz = y1 - y0, z1 - z0
            length = float(np.hypot(dy, dz))
            if length <= 0.0:
                raise ValueError(
                    f"CCX element {elem.eid} on panel {pi}: zero-length edge "
                    f"(nodes {n0}/{n1}); refusing silent length=1.0 (FEA-CCX-004)"
                )
            ty, tz = dy / length, dz / length
            mx, my, mxy = bending_moments_from_frd_ips(
                ip_rows, thickness=float(res.thickness_m or 0.0), ty=ty, tz=tz,
            )
            bounded = _frd_bending_bounded(
                mx, my, mxy, ip_rows, thickness=float(res.thickness_m or 0.0),
            )
            if bounded is None:
                stats["n_rejected_bound"] += 1
                for key in ("M11", "M22", "M12"):
                    res.provenance[key] = FieldProvenance(
                        ProvenanceKind.RESERVED, "frd IP bending failed physical bound",
                    )
                continue
            mx, my, mxy = bounded
            res.M11 = mx
            res.M22 = my
            res.M12 = mxy
            for key in ("M11", "M22", "M12"):
                res.provenance[key] = FieldProvenance(prov_kind, "ccx FRD IP bending")
            stats["n_applied"] += 1
    return stats


@dataclass(frozen=True)
class PeriodicStepResult:
    """Per-mode results from a periodic unit-cell ccx run: element stresses + ref-node reactions."""

    mode: int
    stress_rows: list[CcxElementStress]
    rf_by_node: dict[int, tuple[float, float, float]]


def parse_ccx_dat_periodic(dat_path: str | Path, n_modes: int = 6) -> list[PeriodicStepResult]:
    """Parse a multi-step periodic ``.dat``: per-mode element 3D stresses (``S``) + reference-node
    reactions (``RF``).

    The file is split into successive stress and reaction blocks by their header keywords; the
    *k*-th stress block and *k*-th RF block form mode *k* (the 6 ``*STEP``s are written in mode
    order by :func:`..calculix_ccx.section_export.write_section_inp_periodic`).
    """
    text = Path(dat_path).read_text(encoding="utf-8", errors="replace")
    rf_re = re.compile(r"reaction\s+force|forces\s*\(", re.IGNORECASE)
    nd_row = re.compile(r"^\s*(\d+)\s+([-+eE\d.]+)\s+([-+eE\d.]+)\s+([-+eE\d.]+)\s*$")

    stress_blocks: list[list[str]] = []
    rf_blocks: list[dict[int, tuple[float, float, float]]] = []
    mode: str | None = None  # "s" | "rf" | None
    cur_s: list[str] = []
    cur_rf: dict[int, tuple[float, float, float]] = {}

    def _flush() -> None:
        nonlocal cur_s, cur_rf
        if mode == "s" and cur_s:
            stress_blocks.append(cur_s); cur_s = []
        elif mode == "rf" and cur_rf:
            rf_blocks.append(cur_rf); cur_rf = {}

    for line in text.splitlines():
        if _STRESS_HEADER.search(line):
            _flush(); mode = "s"; continue
        if rf_re.search(line):
            _flush(); mode = "rf"; continue
        if line.strip() == "":
            continue
        if mode == "s":
            cur_s.append(line)
        elif mode == "rf":
            m = nd_row.match(line)
            if m:
                cur_rf[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    _flush()

    steps: list[PeriodicStepResult] = []
    for k in range(min(n_modes, len(stress_blocks))):
        rf = rf_blocks[k] if k < len(rf_blocks) else {}
        steps.append(PeriodicStepResult(mode=k, stress_rows=_parse_stress_block(stress_blocks[k]), rf_by_node=rf))
    return steps


def map_ccx_forces_to_panel_resultants(
    mesh: CcxSectionMesh,
    zones: list[Any],
    force_rows: list[CcxSectionForceRow],
) -> tuple[list[list[ShellPanelResultants]], list[dict]]:
    """Map parsed ccx section forces to per-panel ``ShellPanelResultants``."""
    by_eid = {r.eid: r for r in force_rows}
    prov_kind = ProvenanceKind.CALCULIX_CCX
    all_results: list[list[ShellPanelResultants]] = []
    all_diag: list[dict] = []

    for pi, pm in enumerate(mesh.panel_maps):
        if len(pm.s_nodes) == 0:
            all_results.append([])
            all_diag.append({})
            continue
        p = zones[pi]
        thickness = float(p.lam.t)
        panel_elems = sorted(
            (e for e in mesh.elements if e.panel_index == pi),
            key=lambda e: e.local_element_index,
        )
        results: list[ShellPanelResultants] = []
        for elem in panel_elems:
            row = by_eid.get(elem.eid)
            if row is None:
                results.append(
                    ShellPanelResultants(
                        N11=0.0,
                        N22=0.0,
                        N12=0.0,
                        M11=0.0,
                        M22=0.0,
                        M12=0.0,
                        Q1=0.0,
                        Q2=0.0,
                        provenance={
                            k: FieldProvenance(ProvenanceKind.PLACEHOLDER, "ccx SF missing")
                            for k in ("N11", "N22", "N12", "M11", "M22", "M12", "Q1", "Q2")
                        },
                        thickness_m=thickness,
                        panel_label=pm.panel_label,
                        panel_index=pi,
                        station_index=elem.local_element_index,
                    )
                )
                continue
            results.append(
                ShellPanelResultants(
                    N11=float(row.n11),
                    N22=float(row.n22),
                    N12=float(row.n12),
                    M11=float(row.m11),
                    M22=float(row.m22),
                    M12=float(row.m12),
                    Q1=float(row.q1),
                    Q2=float(row.q2),
                    provenance={
                        k: FieldProvenance(prov_kind, f"ccx SF {k}")
                        for k in ("N11", "N22", "N12", "M11", "M22", "M12", "Q1", "Q2")
                    },
                    sigma_11_pa=float(row.n11) / max(thickness, 1e-30),
                    tau_12_pa=float(row.n12),
                    q_n_per_m=float(row.n12),
                    thickness_m=thickness,
                    panel_label=pm.panel_label,
                    panel_index=pi,
                    station_index=elem.local_element_index,
                )
            )
        all_results.append(results)
        all_diag.append(
            {
                "solver": "global_coupled_ccx",
                "element_kernel": "calculix_ccx",
                "n_parsed_forces": sum(1 for e in mesh.elements if e.panel_index == pi and e.eid in by_eid),
            }
        )

    return all_results, all_diag


def parse_ccx_dat_node_reactions(
    dat_path: str | Path,
    *,
    nset_hint: str | None = None,
) -> dict[int, tuple[float, float, float]]:
    """Parse nodal reaction forces (``RF``) from a single-step CalculiX ``.dat``.

    When ``nset_hint`` is given, only the reaction block whose header mentions that
    nset name (case-insensitive) is kept. Otherwise the last RF block in the file
    is returned.
    """
    text = Path(dat_path).read_text(encoding="utf-8", errors="replace")
    rf_re = re.compile(r"reaction\s+force|forces\s*\(", re.IGNORECASE)
    nd_row = re.compile(r"^\s*(\d+)\s+([-+eE\d.]+)\s+([-+eE\d.]+)\s+([-+eE\d.]+)\s*$")

    blocks: list[tuple[str, dict[int, tuple[float, float, float]]]] = []
    header = ""
    mode = False
    cur: dict[int, tuple[float, float, float]] = {}

    def _flush() -> None:
        nonlocal cur, header, mode
        if cur:
            blocks.append((header, cur))
            cur = {}
        mode = False

    for line in text.splitlines():
        if rf_re.search(line):
            _flush()
            header = line
            mode = True
            continue
        if not mode:
            continue
        # Blank lines often follow the header in ccx .dat; keep collecting until
        # the next block header (do not flush on empty).
        if line.strip() == "":
            continue
        m = nd_row.match(line)
        if m:
            cur[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
            continue
        # Non-matching non-empty line ends the block (e.g. next section header).
        if not _STRESS_HEADER.search(line):
            _flush()
    _flush()

    if not blocks:
        return {}
    if nset_hint:
        hint = nset_hint.upper()
        for hdr, blk in blocks:
            if hint in hdr.upper():
                return blk
    return blocks[-1][1]


@dataclass
class RootSegmentMappedResultants:
    """Resultants from a multi-row root-segment CCX solve.

    ``by_panel_row[pi][span_row]`` is the contour-element resultant list for that panel
    at the given spanwise element row (``0`` = root-adjacent).
    """

    by_panel_row: list[list[list[ShellPanelResultants]]]
    root_row: list[list[ShellPanelResultants]]
    decay_profile: list[dict[str, float]]


def _element_membrane_from_stress(
    *,
    r: CcxElementStress | None,
    mesh_nodes: dict[int, tuple[float, float, float]],
    node_ids: list[int],
    thickness: float,
    panel_label: str,
    panel_index: int,
    station_index: int,
) -> ShellPanelResultants:
    """Project mid-surface 3D stress → wall resultants (same convention as strip mapper)."""
    prov_kind = ProvenanceKind.CALCULIX_CCX
    keys = ("N11", "N22", "N12", "Q1", "Q2")
    n0, n1 = node_ids[0], node_ids[1]
    (_x0, y0, z0) = mesh_nodes[n0]
    (_x1, y1, z1) = mesh_nodes[n1]
    dy, dz = y1 - y0, z1 - z0
    length = float(np.hypot(dy, dz))
    if length <= 0.0:
        raise ValueError(
            f"CCX panel {panel_label!r} station {station_index}: zero-length edge "
            f"(nodes {n0}/{n1}); refusing silent length=1.0 (FEA-CCX-004)"
        )
    ty, tz = dy / length, dz / length
    ny, nz = -tz, ty
    if r is None:
        return ShellPanelResultants(
            N11=0.0, N22=0.0, N12=0.0, M11=0.0, M22=0.0, M12=0.0, Q1=0.0, Q2=0.0,
            provenance={
                k: FieldProvenance(ProvenanceKind.PLACEHOLDER, "ccx S missing")
                for k in ("N11", "N22", "N12", "M11", "M22", "M12", "Q1", "Q2")
            },
            thickness_m=thickness, panel_label=panel_label, panel_index=panel_index,
            station_index=station_index,
        )
    nx = r.sxx * thickness
    nxy = (r.sxy * ty + r.sxz * tz) * thickness
    ny_res = (r.syy * ty * ty + 2.0 * r.syz * ty * tz + r.szz * tz * tz) * thickness
    qx = (r.sxy * ny + r.sxz * nz) * thickness
    qy = ((r.syy * ny + r.syz * nz) * ty + (r.syz * ny + r.szz * nz) * tz) * thickness
    prov = {k: FieldProvenance(prov_kind, f"ccx S->{k}") for k in keys}
    # Diagnostic path: trust M=0 so membrane Hashin can run without FRD. Callers that
    # request emit_frd should upgrade M11/M22/M12 via apply_frd_bending_to_panel_resultants
    # (strip helper) after flattening by span row if needed.
    prov.update({
        k: FieldProvenance(
            prov_kind, "root-segment diagnostic: M=0 mid-surface (FRD optional)"
        )
        for k in ("M11", "M22", "M12")
    })
    return ShellPanelResultants(
        N11=nx, N22=ny_res, N12=nxy, M11=0.0, M22=0.0, M12=0.0, Q1=qx, Q2=qy,
        provenance=prov,
        sigma_11_pa=r.sxx, tau_12_pa=nxy, q_n_per_m=nxy, thickness_m=thickness,
        panel_label=panel_label, panel_index=panel_index, station_index=station_index,
    )


def map_root_segment_stresses_to_resultants(
    mesh: Any,
    zones: list[Any],
    stress_rows: list[CcxElementStress],
) -> RootSegmentMappedResultants:
    """Map multi-row root-segment stresses → per-(panel, span_row) resultants + decay profile.

    ``mesh`` is a :class:`~.root_segment_mesh.CcxRootSegmentMesh`. The root-adjacent
    row (``span_row == 0``) is the payload for station-0 comparison; the decay
    profile tracks max |N11|, |N12|, |N22| per spanwise row toward the outboard cut.
    """
    by_eid = {r.eid: r for r in stress_rows}
    n_rows = int(getattr(mesh, "n_span_rows", 0) or 0)
    n_panels = len(zones)
    by_panel_row: list[list[list[ShellPanelResultants]]] = [
        [[] for _ in range(n_rows)] for _ in range(n_panels)
    ]

    # Group elements by (panel, span_row), sort by contour index
    buckets: dict[tuple[int, int], list[Any]] = {}
    for elem in mesh.elements:
        buckets.setdefault((elem.panel_index, elem.span_row), []).append(elem)

    for (pi, row), elems in buckets.items():
        if pi >= n_panels or row >= n_rows:
            continue
        elems_sorted = sorted(elems, key=lambda e: e.contour_element_index)
        thickness = float(zones[pi].lam.t)
        label = mesh.panel_maps[pi].panel_label if pi < len(mesh.panel_maps) else f"panel_{pi}"
        rows_out: list[ShellPanelResultants] = []
        for elem in elems_sorted:
            rows_out.append(
                _element_membrane_from_stress(
                    r=by_eid.get(elem.eid),
                    mesh_nodes=mesh.nodes,
                    node_ids=elem.node_ids,
                    thickness=thickness,
                    panel_label=label,
                    panel_index=pi,
                    station_index=elem.contour_element_index,
                )
            )
        by_panel_row[pi][row] = rows_out

    root_row = [by_panel_row[pi][0] if n_rows else [] for pi in range(n_panels)]

    decay_profile: list[dict[str, float]] = []
    span = np.asarray(getattr(mesh, "span_coords", []), dtype=float)
    for row in range(n_rows):
        max_nx = max_nxy = max_ny = 0.0
        for pi in range(n_panels):
            for res in by_panel_row[pi][row]:
                max_nx = max(max_nx, abs(float(res.N11)))
                max_nxy = max(max_nxy, abs(float(res.N12)))
                max_ny = max(max_ny, abs(float(res.N22)))
        # Element-row mid-span station (average of bounding node layers)
        if span.size >= row + 2:
            x_mid = 0.5 * (float(span[row]) + float(span[row + 1]))
        elif span.size:
            x_mid = float(span[min(row, span.size - 1)])
        else:
            x_mid = float(row)
        decay_profile.append(
            {
                "span_row": float(row),
                "x_m": x_mid,
                "max_abs_N11": max_nx,
                "max_abs_N12": max_nxy,
                "max_abs_N22": max_ny,
                # Legacy aliases (pre-E-frame CSV consumers).
                "max_abs_Nx": max_nx,
                "max_abs_Nxy": max_nxy,
                "max_abs_Ny": max_ny,
            }
        )

    return RootSegmentMappedResultants(
        by_panel_row=by_panel_row,
        root_row=root_row,
        decay_profile=decay_profile,
    )


__all__ = [
    "CcxSectionForceRow",
    "CcxElementStress",
    "CcxFrdIpStress",
    "FrdCcxNodeIndex",
    "PeriodicStepResult",
    "RootSegmentMappedResultants",
    "apply_frd_bending_to_panel_resultants",
    "bending_moments_from_frd_ips",
    "ccx_frd_element_base",
    "parse_ccx_frd_3c_node_index",
    "parse_ccx_frd_element_map",
    "parse_ccx_frd_stress_blocks",
    "list_unmapped_frd_entities",
    "parse_ccx_dat_section_forces",
    "parse_ccx_dat_stresses",
    "parse_ccx_dat_periodic",
    "parse_ccx_dat_node_reactions",
    "map_ccx_forces_to_panel_resultants",
    "map_ccx_stresses_to_panel_resultants",
    "map_root_segment_stresses_to_resultants",
    "superpose_frd_stress_blocks",
]
