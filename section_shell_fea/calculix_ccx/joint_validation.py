"""MITC4 ↔ CalculiX joint-modeling validation (shared nodes + cap ``*TIE``).

Validation-only. Does not change production MITC4 defaults. Compares section
integrals / interior peaks / load share — not elemental Hashin maxima on
tie-slave or Defect-K corner bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.cap_tie_constraints import (
    _kind_of,
)

from .mesh import build_ccx_section_mesh
from .section_export import write_section_inp

# Relative band for primary section-integral style checks (historical MITC4↔CCX ~10%).
DEFAULT_AGREE_BAND = 0.10


@dataclass
class JointDeckCheck:
    """Offline assertions on a written ``.inp``."""

    path: Path
    has_tie: bool
    n_tie_cards: int
    n_surfaces: int
    shared_junction_ok: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class KindPeakSummary:
    kind: str
    max_abs_nxy: float
    interior_max_abs_nxy: float
    n_elements: int


@dataclass
class JointCompareRow:
    case: str
    layout: str
    cap_tie_mode: str
    mitc4: dict[str, KindPeakSummary] = field(default_factory=dict)
    ccx: dict[str, KindPeakSummary] | None = None
    cap_nxy_share_mitc4: float = float("nan")
    cap_nxy_share_ccx: float = float("nan")
    notes: list[str] = field(default_factory=list)
    live_ccx: bool = False


@dataclass
class JointValidationReport:
    deck_checks: list[JointDeckCheck] = field(default_factory=list)
    compares: list[JointCompareRow] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["CalculiX joint validation"]
        for d in self.deck_checks:
            lines.append(
                f"  deck {d.path.name}: tie={d.has_tie} "
                f"(*TIE={d.n_tie_cards}, *SURFACE={d.n_surfaces}) "
                f"shared_junction_ok={d.shared_junction_ok}"
            )
        for c in self.compares:
            lines.append(
                f"  compare {c.case}/{c.layout} cap_tie={c.cap_tie_mode} "
                f"cap_share_mitc4={c.cap_nxy_share_mitc4:.3f} "
                f"cap_share_ccx={c.cap_nxy_share_ccx:.3f} live={c.live_ccx}"
            )
            for n in c.notes:
                lines.append(f"    note: {n}")
        return "\n".join(lines)


def _interior_mask(n_el: int) -> np.ndarray:
    """Drop endpoint-adjacent elements (Defect K / tie-slave bands)."""
    mask = np.ones(n_el, dtype=bool)
    if n_el <= 2:
        return mask
    mask[0] = False
    mask[-1] = False
    if n_el >= 4:
        mask[1] = False
        mask[-2] = False
    return mask


def summarize_kind_peaks(
    zones: list[Any],
    panel_result_lists: list[list[Any]],
) -> dict[str, KindPeakSummary]:
    """Peak |N12| per kind from panel resultant lists (MITC4 or CCX-mapped)."""
    by_kind: dict[str, list[float]] = {}
    by_kind_int: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for p, res_list in zip(zones, panel_result_lists):
        kind = _kind_of(str(getattr(p, "label", "panel")))
        vals = [abs(float(getattr(r, "N12", 0.0))) for r in (res_list or [])]
        if not vals:
            continue
        by_kind.setdefault(kind, []).extend(vals)
        mask = _interior_mask(len(vals))
        by_kind_int.setdefault(kind, []).extend(
            v for v, keep in zip(vals, mask) if keep
        )
        counts[kind] = counts.get(kind, 0) + len(vals)
    out: dict[str, KindPeakSummary] = {}
    for kind, vals in by_kind.items():
        iv = by_kind_int.get(kind) or vals
        out[kind] = KindPeakSummary(
            kind=kind,
            max_abs_nxy=float(max(vals)),
            interior_max_abs_nxy=float(max(iv)),
            n_elements=int(counts.get(kind, 0)),
        )
    return out


def cap_nxy_share(peaks: dict[str, KindPeakSummary]) -> float:
    """``|N12|_cap / (|N12|_cap + |N12|_skin)`` using interior peaks."""
    cap = peaks.get("cap")
    skin = peaks.get("skin")
    if cap is None or skin is None:
        return float("nan")
    a = float(cap.interior_max_abs_nxy)
    b = float(skin.interior_max_abs_nxy)
    denom = a + b
    if denom <= 0.0:
        return float("nan")
    return a / denom


def inspect_joint_deck(
    inp_path: Path,
    *,
    expect_tie: bool,
    mesh: Any | None = None,
) -> JointDeckCheck:
    """Offline keyword inspection of a section ``.inp``."""
    text = Path(inp_path).read_text(encoding="utf-8")
    n_tie = text.count("*TIE")
    n_surf = text.count("*SURFACE")
    has_tie = n_tie > 0
    notes: list[str] = []
    if expect_tie and not has_tie:
        notes.append("expected *TIE cards missing")
    if not expect_tie and has_tie:
        notes.append("unexpected *TIE cards present")

    shared_ok = True
    if mesh is not None and getattr(mesh, "clusters", None):
        # Each cluster should map endpoint globals of members to the same CCX id.
        for c in mesh.clusters:
            if len(c) < 2:
                continue
            ccx_ids: set[int] = set()
            for pi, which, _ in c:
                pm = mesh.panel_maps[pi]
                n = len(pm.s_nodes)
                if n == 0:
                    continue
                li = 0 if which == "start" else n - 1
                gn = int(pm.global_nodes[li])
                cid = mesh.global_to_ccx_node.get(gn)
                if cid is not None:
                    ccx_ids.add(int(cid))
            if len(ccx_ids) > 1:
                shared_ok = False
                notes.append(f"cluster endpoints not shared: {ccx_ids}")
                break

    return JointDeckCheck(
        path=Path(inp_path),
        has_tie=has_tie,
        n_tie_cards=n_tie,
        n_surfaces=n_surf,
        shared_junction_ok=shared_ok,
        notes=notes,
    )


def write_joint_deck(
    zones: list[Any],
    nx_panels: list[np.ndarray],
    nxy_panels: list[np.ndarray],
    *,
    section_strip_mesh: Any,
    path: str | Path,
    interface_constraint_mode: str = "shared",
    cap_tie_mode: str | None = None,
    cap_tie_export: str = "tie",
    l_x: float = 1.0,
    bc_mode: str = "contour_support",
) -> tuple[Path, Any, JointDeckCheck]:
    """Build mesh, write ``.inp``, return path/mesh/offline check."""
    mesh = build_ccx_section_mesh(
        zones,
        section_strip_mesh=section_strip_mesh,
        interface_constraint_mode=interface_constraint_mode,
        l_x=l_x,
        cap_tie_mode=cap_tie_mode,
    )
    out = write_section_inp(
        mesh,
        zones,
        nx_panels,
        nxy_panels,
        path,
        bc_mode=bc_mode,
        cap_tie_mode=mesh.cap_tie_mode,
        cap_tie_export=cap_tie_export,
    )
    expect = mesh.cap_tie_mode == "skin_offset" and bool(mesh.cap_skin_ties)
    check = inspect_joint_deck(out, expect_tie=expect and cap_tie_export == "tie", mesh=mesh)
    return out, mesh, check


def _synthetic_loads(zones: list[Any]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    nx: list[np.ndarray] = []
    nxy: list[np.ndarray] = []
    for p in zones:
        s = np.asarray(getattr(p, "s"), dtype=float)
        kind = _kind_of(str(getattr(p, "label", "")))
        if kind == "web":
            nx.append(np.zeros_like(s))
            nxy.append(np.full_like(s, 200.0))
        elif kind == "cap":
            nx.append(np.full_like(s, 500.0))
            nxy.append(np.full_like(s, 150.0))
        else:
            nx.append(np.full_like(s, 1000.0))
            nxy.append(np.full_like(s, 50.0))
    return nx, nxy


def build_l_junction_fixture(n_elements_per_panel: int = 4):
    """Minimal skin|web L for shared-node joint checks."""
    from blade_structure_utilities.mesh_utilities.section_mesh import (
        build_section_strip_mesh_from_panels,
    )

    class _Lam:
        E = 70e9
        nu = 0.33
        t = 0.01

    class _Panel:
        def __init__(self, nodes, s, label):
            self.nodes = np.asarray(nodes, dtype=float)
            self.s = np.asarray(s, dtype=float)
            self.label = label
            self.lam = _Lam()

    zones = [
        _Panel([[0, 0], [1, 0]], [0, 1], "skin:a"),
        _Panel([[1, 0], [1, 1]], [0, 1], "web:b"),
    ]
    mesh = build_section_strip_mesh_from_panels(
        zones, n_elements_per_panel=n_elements_per_panel
    )
    nx, nxy = _synthetic_loads(zones)
    return zones, nx, nxy, mesh


def build_capped_catalogue_fixture(
    layout: str = "3D-F",
    *,
    target_element_length_frac: float = 0.02,
):
    """Catalogue section with spar caps (skin_offset joint target).

    Returns ``(zones, nx, nxy, strip_mesh, shell_inputs)``.
    """
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (
        build_section_view,
        resolve_system_type,
    )
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import (
        AirfoilSDF,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh.inputs._section_to_shell_inputs import (
        shell_mesh_inputs_from_section,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh import (
        build_section_strip_mesh,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
        shell_panel_topology_from_shell_inputs,
    )

    sys_type = resolve_system_type(layout)
    if not bool(getattr(sys_type, "has_spar_caps", False)):
        raise ValueError(
            f"layout {layout!r} has no spar caps; use a D-family key (e.g. 3D-F, 5D-F)"
        )
    twist = float(np.deg2rad(34.1))
    airfoil = AirfoilSDF.from_naca_series(
        6, 63.0, 4.0, 15.0, n_points=160, chord=0.241, closed_te=True
    )
    section = build_section_view(
        airfoil,
        sys_type,
        twist_angle_rad=twist,
        skin_thickness=0.001,
        web_thickness=0.0015,
        cap_height=0.004,
    )
    shell = shell_mesh_inputs_from_section(
        section, twist_rad=twist, layout_key=layout, n_web_samples=16, n_cap_samples=40
    )
    strip = build_section_strip_mesh(
        shell, target_element_length_frac=target_element_length_frac, endpoint_tol=1e-6
    )
    zones = list(shell_panel_topology_from_shell_inputs(shell).zones)
    nx, nxy = _synthetic_loads(zones)
    return zones, nx, nxy, strip, shell


def run_mitc4_kind_peaks_from_shell(
    shell: Any,
    section_strip_mesh: Any,
    *,
    cap_tie_mode: str,
    interface_constraint_mode: str = "shared",
    load: dict[str, float] | None = None,
) -> dict[str, KindPeakSummary]:
    """Solve MITC4 cascade-style and return per-kind interior |N12| peaks."""
    from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
        BeamResultantVector,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
        shell_panel_topology_from_shell_inputs,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
        SectionLinearStaticRunner,
    )
    from blade_structure_utilities.load_utilities.section_loads import (
        shape_section_force_vector,
    )

    load = dict(load or {"T": 5.0e3, "Vy": 2.0e4, "Mz": 1.0e4})
    topo = shell_panel_topology_from_shell_inputs(shell)
    resultants = BeamResultantVector.from_named(**load)
    runner = SectionLinearStaticRunner(
        panel_topology=topo,
        section_strip_mesh=section_strip_mesh,
        resultants=resultants,
        section_force_vector=shape_section_force_vector(resultants, shell.midlines),
        load_channel="thinwall",
        cap_tie_mode=cap_tie_mode,
        interface_constraint_mode=interface_constraint_mode,
    )
    runner.execute()
    results = runner.get_results()
    zones = list(topo.zones)
    n_panel = len(zones)
    res_lists: list[list[Any]] = [[] for _ in range(n_panel)]
    sec = results.secondary_results
    if sec is not None and sec.elemental_results is not None:
        for row in sec.elemental_results.resultant_rows or []:
            pi = int(row.panel_index)
            if 0 <= pi < n_panel:
                res_lists[pi].append(row)
    return summarize_kind_peaks(zones, res_lists)


def run_joint_validation(
    work_dir: str | Path,
    *,
    layouts: tuple[str, ...] = ("3D-F",),
    run_live_ccx: bool = False,
    run_mitc4: bool = True,
    ccx_exe: str = "ccx",
) -> JointValidationReport:
    """Write joint decks + optional MITC4/CCX compare for L-junction and capped layouts."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    report = JointValidationReport()

    # --- Skin–web L (shared nodes, no *TIE) ---
    zones, nx, nxy, strip = build_l_junction_fixture()
    _path, mesh, check = write_joint_deck(
        zones,
        nx,
        nxy,
        section_strip_mesh=strip,
        path=work / "l_junction_shared.inp",
        cap_tie_mode="none",
    )
    report.deck_checks.append(check)
    row = JointCompareRow(case="skin_web_L", layout="L", cap_tie_mode="none")
    if not check.shared_junction_ok:
        row.notes.append("L-junction shared nodes failed offline check")
    if check.has_tie:
        row.notes.append("L-junction deck unexpectedly contains *TIE")
    if run_live_ccx:
        try:
            from .solve import solve_global_coupled_ccx

            results, _diag, _ = solve_global_coupled_ccx(
                zones,
                nx,
                nxy,
                section_strip_mesh=strip,
                work_dir=work / "ccx_l",
                ccx_exe=ccx_exe,
                cap_tie_mode="none",
                job_name="l_junction",
            )
            row.ccx = summarize_kind_peaks(zones, results)
            row.live_ccx = True
        except Exception as exc:  # noqa: BLE001 — validation harness
            row.notes.append(f"live CCX L-junction failed: {exc}")
    report.compares.append(row)

    # --- Skin–cap offset (*TIE) ---
    for layout in layouts:
        zones, nx, nxy, strip, shell = build_capped_catalogue_fixture(layout)
        mesh_last = None
        for mode in ("skin_offset", "none"):
            _path, mesh, check = write_joint_deck(
                zones,
                nx,
                nxy,
                section_strip_mesh=strip,
                path=work / f"{layout}_{mode}.inp",
                cap_tie_mode=mode,
            )
            mesh_last = mesh
            report.deck_checks.append(check)
            crow = JointCompareRow(case="skin_cap", layout=layout, cap_tie_mode=mode)
            if run_mitc4:
                try:
                    crow.mitc4 = run_mitc4_kind_peaks_from_shell(
                        shell, strip, cap_tie_mode=mode
                    )
                    crow.cap_nxy_share_mitc4 = cap_nxy_share(crow.mitc4)
                except Exception as exc:  # noqa: BLE001
                    crow.notes.append(f"MITC4 solve failed: {exc}")
            if mode == "skin_offset" and not mesh.cap_skin_ties:
                crow.notes.append("no cap_skin_ties built for catalogue layout")
            if mode == "skin_offset" and not check.has_tie:
                crow.notes.append("skin_offset deck missing *TIE")
            if mode == "none" and check.has_tie:
                crow.notes.append("untied deck unexpectedly contains *TIE")
            if run_live_ccx and mode == "skin_offset":
                try:
                    from .solve import solve_global_coupled_ccx

                    results, _diag, _ = solve_global_coupled_ccx(
                        zones,
                        nx,
                        nxy,
                        section_strip_mesh=strip,
                        work_dir=work / f"ccx_{layout}",
                        ccx_exe=ccx_exe,
                        cap_tie_mode="skin_offset",
                        job_name=f"{layout}_tie",
                    )
                    crow.ccx = summarize_kind_peaks(zones, results)
                    crow.cap_nxy_share_ccx = cap_nxy_share(crow.ccx)
                    crow.live_ccx = True
                    s_m = crow.cap_nxy_share_mitc4
                    s_c = crow.cap_nxy_share_ccx
                    if np.isfinite(s_m) and np.isfinite(s_c):
                        rel = abs(s_m - s_c) / max(abs(s_m), 1e-30)
                        if rel > DEFAULT_AGREE_BAND:
                            crow.notes.append(
                                f"cap N12 share disagree rel={rel:.3f} "
                                f"(band={DEFAULT_AGREE_BAND})"
                            )
                except Exception as exc:  # noqa: BLE001
                    crow.notes.append(f"live CCX capped solve failed: {exc}")
            report.compares.append(crow)

        # Cap–web continuity: shared junction clusters involving cap+web.
        if mesh_last is not None:
            cap_web_shared = False
            for c in mesh_last.clusters:
                kinds = {
                    _kind_of(mesh_last.panel_maps[pi].panel_label)
                    for pi, _which, _ in c
                }
                if "cap" in kinds and "web" in kinds:
                    cap_web_shared = True
                    break
            if not cap_web_shared:
                report.compares[-1].notes.append(
                    "no cap+web shared cluster on this layout (geometry-dependent)"
                )

    return report


__all__ = [
    "DEFAULT_AGREE_BAND",
    "JointCompareRow",
    "JointDeckCheck",
    "JointValidationReport",
    "KindPeakSummary",
    "build_capped_catalogue_fixture",
    "build_l_junction_fixture",
    "cap_nxy_share",
    "inspect_joint_deck",
    "run_joint_validation",
    "run_mitc4_kind_peaks_from_shell",
    "summarize_kind_peaks",
    "write_joint_deck",
]
