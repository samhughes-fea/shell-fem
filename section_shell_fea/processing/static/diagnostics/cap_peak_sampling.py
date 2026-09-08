"""Locate and characterize kind-wise peaks under mesh refinement.

Compares raw elemental maxima (cascade gate metrics) to robust percentiles and
interior-excluded peaks, and tags the argmax element as junction / tie-slave /
endpoint-band. Diagnose-only: does not change production gates or mesher defaults.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

import numpy as np

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BeamResultantVector,
)
from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (
    resolve_system_type,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    shell_panel_topology_from_shell_inputs,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cascade_mesh_convergence import (
    ENDPOINT_BAND,
    KINDS,
    _build_shell,
    _kind_from_label,
    endpoint_band_mask,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.ply_results import (
    FACE_MID,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.grid_independence_study import (
    PRODUCTION_SUBLADDER,
    STUDY_REFINEMENTS,
    cap_tie_mode_for_layout,
    load_cotide_station_recipe,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)
from blade_structure_utilities.load_utilities.section_loads import (
    shape_section_force_vector,
)
from blade_structure_utilities.mesh_utilities.section_mesh import (
    assert_strip_mesh_quality,
    build_section_strip_mesh,
    mesh_quality_report,
)

ProgressFn = Callable[[str], None]

DEFAULT_INVESTIGATION_KEYS: tuple[str, ...] = (
    "4D-F",
    "5D-F",
    "0A",
    "3B-F",
    "3B-CN",
    "4D-CN",
    "2D-F",
)
METRICS = (
    "peak_abs_n12",
    "peak_abs_m11",
    "peak_abs_m22",
    "peak_abs_q2",
    "peak_mid_sig",
    "peak_hashin",
    "peak_hashin_il",
)
STATISTICS = ("max", "p50", "p95", "p99", "interior_max", "interface_excluded_max")
SAMPLING_DRIFT_LIMIT = 2.0
ROBUST_STABLE_LIMIT = 1.25


@dataclass(frozen=True)
class ElementPeakTags:
    """Boolean tags for one peaking element."""

    junction: bool = False
    tie_slave: bool = False
    endpoint_band: bool = False

    @property
    def any_interface(self) -> bool:
        return bool(self.junction or self.tie_slave or self.endpoint_band)


@dataclass(frozen=True)
class PeakLocation:
    """Argmax of one metric for one kind at one mesh density."""

    kind: str
    metric: str
    value: float
    panel_index: int
    panel_label: str
    local_element_index: int
    ply_index: int = -1
    s_m: float = float("nan")
    tags: ElementPeakTags = field(default_factory=ElementPeakTags)


@dataclass(frozen=True)
class KindMetricStats:
    """Raw / robust / interior peaks for one kind × metric."""

    kind: str
    metric: str
    n_elem: int
    max: float = float("nan")
    p50: float = float("nan")
    p95: float = float("nan")
    p99: float = float("nan")
    interior_max: float = float("nan")
    interface_excluded_max: float = float("nan")
    location: PeakLocation | None = None


@dataclass(frozen=True)
class CapPeakRung:
    """Full peak-sampling report for one layout × frac."""

    layout: str
    frac: float
    h_m: float
    mesh_ok: bool
    cascade_trust_ok: bool
    cap_tie_mode: str
    interface_constraint_mode: str
    n_cap_tie_links: int
    stats: tuple[KindMetricStats, ...] = ()


@dataclass(frozen=True)
class CapPeakDriftRow:
    """Drift of one statistic across a ladder segment."""

    layout: str
    kind: str
    metric: str
    statistic: str
    production_drift: float
    last_two_drift: float
    notes: str = ""


def percentile_stats(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """max / p50 / p95 / p99 over finite non-negative samples (NaN if empty)."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "max": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    return {
        "max": float(np.nanmax(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _safe_drift(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v) and float(v) > 0.0]
    if len(finite) != len(values) or not finite:
        return float("nan")
    return max(finite) / min(finite)


def _safe_last_two(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return _safe_drift(values[-2:])


def _kind_from_label(label: object) -> str:
    text = str(label)
    return text.split(":", 1)[0] if ":" in text else "panel"


def _kind_owner_counts(panel_maps: Sequence[Any]) -> dict[int, int]:
    """Distinct component kinds per global node (kind-crossing = junction)."""
    kinds_by_node: dict[int, set[str]] = {}
    for pm in panel_maps:
        kind = _kind_from_label(getattr(pm, "panel_label", ""))
        for gn in set(int(g) for g in pm.global_nodes if int(g) >= 0):
            kinds_by_node.setdefault(gn, set()).add(kind)
    return {gn: len(kinds) for gn, kinds in kinds_by_node.items()}


def _element_global_nodes(pm: Any, local_ei: int) -> list[int]:
    elems = list(pm.elements or [])
    if not (0 <= int(local_ei) < len(elems)):
        return []
    gids: list[int] = []
    for li in elems[int(local_ei)]:
        li_i = int(li)
        if 0 <= li_i < len(pm.global_nodes):
            gn = int(pm.global_nodes[li_i])
            if gn >= 0:
                gids.append(gn)
    return gids


def _element_s_m(pm: Any, local_ei: int) -> float:
    s = np.asarray(pm.s_nodes, dtype=float)
    elems = list(pm.elements or [])
    if s.size < 2 or not (0 <= int(local_ei) < len(elems)):
        return float("nan")
    corners = elems[int(local_ei)]
    # MITC4 strip: local nodes 0,1 are bottom-edge s-nodes of the element.
    i0 = int(corners[0])
    i1 = int(corners[1]) if len(corners) > 1 else i0
    n = len(s)
    # bottom row uses indices 0..n-1; top uses n..2n-1
    si0 = i0 if i0 < n else i0 - n
    si1 = i1 if i1 < n else i1 - n
    if not (0 <= si0 < n and 0 <= si1 < n):
        return float("nan")
    return 0.5 * (float(s[si0]) + float(s[si1]))


def tag_element(
    *,
    n_elem_panel: int,
    local_ei: int,
    element_gids: Sequence[int],
    owner_count: Mapping[int, int],
    tie_slave_nodes: set[int],
    band: int = ENDPOINT_BAND,
) -> ElementPeakTags:
    """Tag one element from ownership / tie / endpoint metadata."""
    ep = bool(endpoint_band_mask(n_elem_panel, band=band)[int(local_ei)]) if n_elem_panel else False
    junction = any(int(owner_count.get(int(g), 0)) > 1 for g in element_gids)
    tie = any(int(g) in tie_slave_nodes for g in element_gids)
    return ElementPeakTags(junction=junction, tie_slave=tie, endpoint_band=ep)


def _default_progress(stream: TextIO = sys.stdout) -> ProgressFn:
    def _emit(msg: str) -> None:
        print(msg, file=stream, flush=True)

    return _emit


def _panel_kind_indices(panel_maps: Sequence[Any]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {k: [] for k in KINDS}
    for pm in panel_maps:
        kind = _kind_from_label(pm.panel_label)
        if kind in out:
            out[kind].append(int(pm.panel_index))
    return out


def analyze_cap_peak_rung(
    frac: float,
    *,
    layout: str,
    chord: float,
    twist_deg: float,
    thickness: Mapping[str, float],
    load: Mapping[str, float],
    naca_dat: str | Path | None = None,
    cap_tie_mode: str | None = None,
    interface_constraint_mode: str = "shared",
    require_mesh_qa: bool = True,
    endpoint_band: int = ENDPOINT_BAND,
    junction_band_elements: int | None = None,
    junction_band_step_over_h: float | None = None,
    junction_band_panel_labels: Sequence[str] | None = None,
    junction_band_which: str = "both",
) -> CapPeakRung:
    """Solve one density rung and return raw / robust / tagged peak stats."""
    tie = cap_tie_mode if cap_tie_mode is not None else cap_tie_mode_for_layout(layout)
    shell = _build_shell(
        layout=layout,
        chord=float(chord),
        twist_deg=float(twist_deg),
        thickness=thickness,
        naca_dat=naca_dat,
    )
    mesh = build_section_strip_mesh(
        shell,
        target_element_length_frac=float(frac),
        junction_band_elements=junction_band_elements,
        junction_band_step_over_h=junction_band_step_over_h,
        junction_band_panel_labels=junction_band_panel_labels,
        junction_band_which=junction_band_which,
    )
    report = mesh_quality_report(mesh, target_element_length_frac=float(frac))
    if require_mesh_qa:
        assert_strip_mesh_quality(mesh, target_element_length_frac=float(frac))

    resultants = BeamResultantVector.from_named(**dict(load))
    runner = SectionLinearStaticRunner(
        panel_topology=shell_panel_topology_from_shell_inputs(shell, allow_demo_bind=True),
        section_strip_mesh=mesh,
        resultants=resultants,
        section_force_vector=shape_section_force_vector(resultants, shell.midlines),
        load_channel="thinwall",
        cap_tie_mode=tie,
        interface_constraint_mode=interface_constraint_mode,
    )
    runner.execute()
    results = runner.get_results()
    asm = runner.assembly
    panel_maps = list(asm.panel_maps)
    owner_count = _kind_owner_counts(panel_maps)
    tie_slaves: set[int] = set()
    n_links = 0
    if runner.cap_tie_report is not None:
        n_links = int(runner.cap_tie_report.n_links)
        tie_slaves = {int(lk.slave_node) for lk in runner.cap_tie_report.links}

    labels = [str(pm.panel_label) for pm in panel_maps]
    n_panel = len(panel_maps)
    n_elem_by_panel = [len(list(pm.elements or [])) for pm in panel_maps]

    # Per-panel elemental fields (aligned to local element index).
    abs_n12 = [np.full(n, np.nan, dtype=float) for n in n_elem_by_panel]
    abs_m11 = [np.full(n, np.nan, dtype=float) for n in n_elem_by_panel]
    abs_m22 = [np.full(n, np.nan, dtype=float) for n in n_elem_by_panel]
    abs_q2 = [np.full(n, np.nan, dtype=float) for n in n_elem_by_panel]
    mid_sig = [np.full(n, np.nan, dtype=float) for n in n_elem_by_panel]
    hashin = [np.full(n, np.nan, dtype=float) for n in n_elem_by_panel]
    hashin_il = [np.full(n, np.nan, dtype=float) for n in n_elem_by_panel]
    hashin_ply = [np.full(n, -1, dtype=int) for n in n_elem_by_panel]

    if results.secondary_results is not None and results.secondary_results.elemental_results:
        for row in results.secondary_results.elemental_results.resultant_rows or []:
            pi = int(row.panel_index)
            ei = int(row.station_index)
            if not (0 <= pi < n_panel and 0 <= ei < n_elem_by_panel[pi]):
                continue
            if row.N12 is not None:
                abs_n12[pi][ei] = abs(float(row.N12))
            if row.M11 is not None:
                abs_m11[pi][ei] = abs(float(row.M11))
            if row.M22 is not None:
                abs_m22[pi][ei] = abs(float(row.M22))
            if row.Q2 is not None:
                abs_q2[pi][ei] = abs(float(row.Q2))

    tertiary = results.tertiary_results
    if tertiary is not None and tertiary.ply_results is not None:
        ply = tertiary.ply_results
        sig = np.asarray(ply.sig_ply_mat, dtype=float)
        mask = (
            np.asarray(ply.trusted_mask, dtype=bool)
            if ply.trusted_mask is not None
            else np.ones(sig.shape[0], dtype=bool)
        )
        panel_index = np.asarray(tertiary.panel_index, dtype=int)
        elem_index = (
            np.asarray(tertiary.element_index, dtype=int)
            if tertiary.element_index is not None
            else np.arange(panel_index.size, dtype=int)
        )
        for gi in range(panel_index.size):
            if not bool(mask[gi]) or sig.size == 0:
                continue
            pi = int(panel_index[gi])
            ei = int(elem_index[gi])
            if not (0 <= pi < n_panel and 0 <= ei < n_elem_by_panel[pi]):
                continue
            block = sig[gi, :, FACE_MID, :]
            mid_sig[pi][ei] = float(np.linalg.norm(block.reshape(-1)))

    if results.quaternary_results is not None:
        h = results.quaternary_results.hashin_fi
        if h is not None and tertiary is not None:
            panel_index = np.asarray(tertiary.panel_index, dtype=int)
            elem_index = (
                np.asarray(tertiary.element_index, dtype=int)
                if tertiary.element_index is not None
                else np.arange(panel_index.size, dtype=int)
            )
            if h.in_plane is not None:
                in_plane = np.asarray(h.in_plane, dtype=float)
                for gi in range(min(panel_index.size, in_plane.shape[0])):
                    pi = int(panel_index[gi])
                    ei = int(elem_index[gi])
                    if not (0 <= pi < n_panel and 0 <= ei < n_elem_by_panel[pi]):
                        continue
                    row = in_plane[gi]
                    if row.ndim == 0:
                        val = float(row)
                        ply_i = 0
                    else:
                        if not np.any(np.isfinite(row)):
                            continue
                        ply_i = int(np.nanargmax(row))
                        val = float(row[ply_i])
                    prev = hashin[pi][ei]
                    if not np.isfinite(prev) or val > prev:
                        hashin[pi][ei] = val
                        hashin_ply[pi][ei] = ply_i
            if h.interlaminar is not None:
                il = np.asarray(h.interlaminar, dtype=float).ravel()
                for gi in range(min(panel_index.size, il.size)):
                    pi = int(panel_index[gi])
                    ei = int(elem_index[gi])
                    if not (0 <= pi < n_panel and 0 <= ei < n_elem_by_panel[pi]):
                        continue
                    val = abs(float(il[gi]))
                    prev = hashin_il[pi][ei]
                    if not np.isfinite(prev) or val > prev:
                        hashin_il[pi][ei] = val

    field_by_metric = {
        "peak_abs_n12": abs_n12,
        "peak_abs_m11": abs_m11,
        "peak_abs_m22": abs_m22,
        "peak_abs_q2": abs_q2,
        "peak_mid_sig": mid_sig,
        "peak_hashin": hashin,
        "peak_hashin_il": hashin_il,
    }
    kind_panels = _panel_kind_indices(panel_maps)
    stats_out: list[KindMetricStats] = []

    for kind in KINDS:
        panels = kind_panels.get(kind, [])
        for metric in METRICS:
            per_panel = field_by_metric[metric]
            values: list[float] = []
            value_meta: list[tuple[int, int]] = []  # (pi, ei)
            interior_vals: list[float] = []
            iface_excl_vals: list[float] = []
            for pi in panels:
                arr = per_panel[pi]
                n = int(arr.size)
                ep_mask = endpoint_band_mask(n, band=endpoint_band)
                for ei in range(n):
                    v = float(arr[ei])
                    if not np.isfinite(v):
                        continue
                    values.append(v)
                    value_meta.append((pi, ei))
                    gids = _element_global_nodes(panel_maps[pi], ei)
                    tags = tag_element(
                        n_elem_panel=n,
                        local_ei=ei,
                        element_gids=gids,
                        owner_count=owner_count,
                        tie_slave_nodes=tie_slaves,
                        band=endpoint_band,
                    )
                    if not ep_mask[ei]:
                        interior_vals.append(v)
                    if not tags.any_interface:
                        iface_excl_vals.append(v)

            pct = percentile_stats(values)
            loc: PeakLocation | None = None
            if values:
                imax = int(np.argmax(np.asarray(values, dtype=float)))
                pi_star, ei_star = value_meta[imax]
                pm = panel_maps[pi_star]
                gids = _element_global_nodes(pm, ei_star)
                tags = tag_element(
                    n_elem_panel=n_elem_by_panel[pi_star],
                    local_ei=ei_star,
                    element_gids=gids,
                    owner_count=owner_count,
                    tie_slave_nodes=tie_slaves,
                    band=endpoint_band,
                )
                ply_i = int(hashin_ply[pi_star][ei_star]) if metric == "peak_hashin" else -1
                loc = PeakLocation(
                    kind=kind,
                    metric=metric,
                    value=float(values[imax]),
                    panel_index=int(pi_star),
                    panel_label=str(labels[pi_star]),
                    local_element_index=int(ei_star),
                    ply_index=ply_i,
                    s_m=_element_s_m(pm, ei_star),
                    tags=tags,
                )
            n_kind = int(sum(n_elem_by_panel[pi] for pi in panels))
            stats_out.append(
                KindMetricStats(
                    kind=kind,
                    metric=metric,
                    n_elem=n_kind,
                    max=pct["max"],
                    p50=pct["p50"],
                    p95=pct["p95"],
                    p99=pct["p99"],
                    interior_max=float(np.nanmax(interior_vals)) if interior_vals else float("nan"),
                    interface_excluded_max=(
                        float(np.nanmax(iface_excl_vals)) if iface_excl_vals else float("nan")
                    ),
                    location=loc,
                )
            )

    diag = results.diagnostics
    return CapPeakRung(
        layout=str(layout),
        frac=float(frac),
        h_m=float(report.h_m),
        mesh_ok=bool(report.ok),
        cascade_trust_ok=bool(diag.cascade_trust_ok) if diag is not None else True,
        cap_tie_mode=str(tie),
        interface_constraint_mode=str(interface_constraint_mode),
        n_cap_tie_links=n_links,
        stats=tuple(stats_out),
    )


def run_cap_peak_ladder(
    layout: str,
    *,
    refinements: Sequence[float] = STUDY_REFINEMENTS,
    progress: ProgressFn | None = None,
    **kwargs: Any,
) -> list[CapPeakRung]:
    """Density ladder of :func:`analyze_cap_peak_rung` for one layout."""
    rows: list[CapPeakRung] = []
    n_rung = len(tuple(refinements))
    for ri, frac in enumerate(refinements, start=1):
        t0 = time.perf_counter()
        if progress is not None:
            progress(f"    rung {ri}/{n_rung}  frac={float(frac):.4f}  ...")
        row = analyze_cap_peak_rung(float(frac), layout=layout, **kwargs)
        rows.append(row)
        if progress is not None:
            cap_h = next(
                (s for s in row.stats if s.kind == "cap" and s.metric == "peak_hashin"),
                None,
            )
            if cap_h is not None and cap_h.location is not None:
                loc = cap_h.location
                tags = loc.tags
                progress(
                    f"    rung {ri}/{n_rung}  done in {time.perf_counter() - t0:.1f}s  "
                    f"cap_hashin_max={cap_h.max:.4g}  p95={cap_h.p95:.4g}  "
                    f"interior={cap_h.interior_max:.4g}  "
                    f"at {loc.panel_label}[e{loc.local_element_index}]  "
                    f"tags=j{int(tags.junction)}/t{int(tags.tie_slave)}/e{int(tags.endpoint_band)}"
                )
            else:
                progress(
                    f"    rung {ri}/{n_rung}  done in {time.perf_counter() - t0:.1f}s  "
                    f"(no cap hashin)"
                )
    return rows


def _stat_series(
    rows: Sequence[CapPeakRung], *, kind: str, metric: str, statistic: str
) -> list[float]:
    out: list[float] = []
    for row in rows:
        hit = next((s for s in row.stats if s.kind == kind and s.metric == metric), None)
        if hit is None:
            out.append(float("nan"))
            continue
        out.append(float(getattr(hit, statistic)))
    return out


def verdict_notes(
    *,
    max_prod: float,
    max_lt: float,
    p95_prod: float,
    p95_lt: float,
    interior_prod: float,
    interior_lt: float,
    location_tags: Sequence[ElementPeakTags],
    interface_excluded_finite: bool = True,
) -> str:
    """Classify sampling artifact vs bulk non-convergence for one series.

    Under ``skin_offset``, every cap element typically contains a tie-slave node,
    so production-ladder p95 can still track the raw max when ``n_elem`` is small.
    Primary robust signal is therefore **last-two** p95 / interior_max, plus
    whether the argmax is interface-tagged.
    """

    def _gt(a: float, lim: float) -> bool:
        return bool(np.isfinite(a) and a > lim)

    def _le(a: float, lim: float) -> bool:
        return bool(np.isfinite(a) and a <= lim)

    max_bad = _gt(max_prod, SAMPLING_DRIFT_LIMIT) or _gt(max_lt, SAMPLING_DRIFT_LIMIT)
    # Prefer last-two robust stability; production p95 often collapses to max on coarse rungs.
    robust_lt_ok = all(
        _le(v, ROBUST_STABLE_LIMIT) or not np.isfinite(v) for v in (p95_lt, interior_lt)
    )
    robust_lt_ci_ok = all(
        _le(v, SAMPLING_DRIFT_LIMIT) or not np.isfinite(v) for v in (p95_lt, interior_lt)
    )
    robust_lt_bad = any(_gt(v, SAMPLING_DRIFT_LIMIT) for v in (p95_lt, interior_lt))
    tagged = any(t.any_interface for t in location_tags)
    endpoint_or_tie = any(t.endpoint_band or t.tie_slave for t in location_tags)
    if not max_bad:
        return "pass"
    if tagged and endpoint_or_tie and (robust_lt_ok or robust_lt_ci_ok):
        suffix = "" if interface_excluded_finite else "|all_cap_elems_tie_tagged"
        if robust_lt_ok:
            return f"sampling_artifact{suffix}"
        return f"sampling_artifact_ci_band{suffix}"
    if robust_lt_bad:
        return "bulk_nonconvergence"
    if tagged:
        return "mixed_or_untagged"
    return "mixed_or_untagged"


def summarize_cap_peak_drifts(rows: Sequence[CapPeakRung]) -> list[CapPeakDriftRow]:
    """Production + last-two drifts for each kind × metric × statistic."""
    if not rows:
        return []
    layout = rows[0].layout
    fracs = [float(r.frac) for r in rows]
    prod_idx = [i for i, f in enumerate(fracs) if float(f) in set(map(float, PRODUCTION_SUBLADDER))]
    out: list[CapPeakDriftRow] = []
    for kind in KINDS:
        for metric in METRICS:
            loc_tags: list[ElementPeakTags] = []
            for row in rows:
                st = next((s for s in row.stats if s.kind == kind and s.metric == metric), None)
                if st is not None and st.location is not None:
                    loc_tags.append(st.location.tags)
            for statistic in STATISTICS:
                series = _stat_series(rows, kind=kind, metric=metric, statistic=statistic)
                prod_series = [series[i] for i in prod_idx] if prod_idx else series[:3]
                notes = ""
                if statistic == "max":
                    p95 = _stat_series(rows, kind=kind, metric=metric, statistic="p95")
                    interior = _stat_series(
                        rows, kind=kind, metric=metric, statistic="interior_max"
                    )
                    iex = _stat_series(
                        rows,
                        kind=kind,
                        metric=metric,
                        statistic="interface_excluded_max",
                    )
                    notes = verdict_notes(
                        max_prod=_safe_drift(prod_series),
                        max_lt=_safe_last_two(series),
                        p95_prod=_safe_drift([p95[i] for i in prod_idx] if prod_idx else p95[:3]),
                        p95_lt=_safe_last_two(p95),
                        interior_prod=_safe_drift(
                            [interior[i] for i in prod_idx] if prod_idx else interior[:3]
                        ),
                        interior_lt=_safe_last_two(interior),
                        location_tags=loc_tags,
                        interface_excluded_finite=any(np.isfinite(v) for v in iex),
                    )
                out.append(
                    CapPeakDriftRow(
                        layout=layout,
                        kind=kind,
                        metric=metric,
                        statistic=statistic,
                        production_drift=_safe_drift(prod_series),
                        last_two_drift=_safe_last_two(series),
                        notes=notes,
                    )
                )
    return out


def write_cap_peak_ladder_csv(path: str | Path, by_layout: Mapping[str, Sequence[CapPeakRung]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layout",
        "frac",
        "h_m",
        "mesh_ok",
        "cascade_trust_ok",
        "cap_tie_mode",
        "n_cap_tie_links",
        "kind",
        "metric",
        "n_elem",
        *STATISTICS,
        "panel_label",
        "local_element_index",
        "ply_index",
        "s_m",
        "tag_junction",
        "tag_tie_slave",
        "tag_endpoint_band",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for layout, rows in by_layout.items():
            for row in rows:
                for st in row.stats:
                    loc = st.location
                    tags = loc.tags if loc is not None else ElementPeakTags()
                    w.writerow(
                        {
                            "layout": layout,
                            "frac": row.frac,
                            "h_m": row.h_m,
                            "mesh_ok": row.mesh_ok,
                            "cascade_trust_ok": row.cascade_trust_ok,
                            "cap_tie_mode": row.cap_tie_mode,
                            "n_cap_tie_links": row.n_cap_tie_links,
                            "kind": st.kind,
                            "metric": st.metric,
                            "n_elem": st.n_elem,
                            "max": st.max,
                            "p50": st.p50,
                            "p95": st.p95,
                            "p99": st.p99,
                            "interior_max": st.interior_max,
                            "interface_excluded_max": st.interface_excluded_max,
                            "panel_label": "" if loc is None else loc.panel_label,
                            "local_element_index": -1 if loc is None else loc.local_element_index,
                            "ply_index": -1 if loc is None else loc.ply_index,
                            "s_m": float("nan") if loc is None else loc.s_m,
                            "tag_junction": bool(tags.junction),
                            "tag_tie_slave": bool(tags.tie_slave),
                            "tag_endpoint_band": bool(tags.endpoint_band),
                        }
                    )
    return path


def write_cap_peak_locations_json(
    path: str | Path, by_layout: Mapping[str, Sequence[CapPeakRung]]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"layouts": {}}
    for layout, rows in by_layout.items():
        payload["layouts"][layout] = []
        for row in rows:
            rung: dict[str, Any] = {
                "frac": row.frac,
                "h_m": row.h_m,
                "cap_tie_mode": row.cap_tie_mode,
                "n_cap_tie_links": row.n_cap_tie_links,
                "locations": [],
            }
            for st in row.stats:
                if st.location is None:
                    continue
                loc = st.location
                rung["locations"].append(
                    {
                        "kind": loc.kind,
                        "metric": loc.metric,
                        "value": loc.value,
                        "panel_index": loc.panel_index,
                        "panel_label": loc.panel_label,
                        "local_element_index": loc.local_element_index,
                        "ply_index": loc.ply_index,
                        "s_m": loc.s_m,
                        "tags": asdict(loc.tags),
                        "stats": {
                            "max": st.max,
                            "p50": st.p50,
                            "p95": st.p95,
                            "p99": st.p99,
                            "interior_max": st.interior_max,
                            "interface_excluded_max": st.interface_excluded_max,
                        },
                    }
                )
            payload["layouts"][layout].append(rung)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_cap_peak_drift_summary_csv(
    path: str | Path, drift_rows: Sequence[CapPeakDriftRow]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layout",
        "kind",
        "metric",
        "statistic",
        "production_drift",
        "last_two_drift",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in drift_rows:
            w.writerow(asdict(r))
    return path


def run_cap_peak_study(
    keys: Sequence[str] = DEFAULT_INVESTIGATION_KEYS,
    *,
    refinements: Sequence[float] = STUDY_REFINEMENTS,
    out_dir: str | Path | None = None,
    require_mesh_qa: bool = True,
    progress: bool | ProgressFn = True,
    recipe: Any | None = None,
    interface_constraint_mode: str = "shared",
    junction_band_elements: int | None = None,
    junction_band_step_over_h: float | None = None,
    junction_band_panel_labels: Sequence[str] | None = None,
    junction_band_which: str = "both",
) -> tuple[dict[str, list[CapPeakRung]], list[CapPeakDriftRow]]:
    """Cotide-backed multi-key cap-peak sampling study."""
    log: ProgressFn | None
    if progress is True:
        log = _default_progress()
    elif progress is False or progress is None:
        log = None
    else:
        log = progress

    active = recipe or load_cotide_station_recipe()
    kw = active.cascade_kwargs()
    iface = str(interface_constraint_mode or "shared").strip()
    kw["interface_constraint_mode"] = iface
    if junction_band_elements is not None:
        kw["junction_band_elements"] = int(junction_band_elements)
    if junction_band_step_over_h is not None:
        kw["junction_band_step_over_h"] = float(junction_band_step_over_h)
    if junction_band_panel_labels is not None:
        kw["junction_band_panel_labels"] = tuple(str(x) for x in junction_band_panel_labels)
    if junction_band_which and str(junction_band_which) != "both":
        kw["junction_band_which"] = str(junction_band_which)
    layout_keys = tuple(str(k) for k in keys)
    t0 = time.perf_counter()
    if log is not None:
        log(
            f"[cap-peak] start  layouts={len(layout_keys)}  "
            f"rungs={len(tuple(refinements))}  fracs={list(map(float, refinements))}  "
            f"interface={iface}"
        )
        log(
            f"[cap-peak] station  {active.station_label}  zL={active.zL:.6g}  "
            f"chord={active.chord_m:.6g} m"
        )

    by_layout: dict[str, list[CapPeakRung]] = {}
    drift_rows: list[CapPeakDriftRow] = []
    for i, key in enumerate(layout_keys, start=1):
        t_layout = time.perf_counter()
        if log is not None:
            log(
                f"[cap-peak] [{i}/{len(layout_keys)}] {key}  "
                f"family={resolve_system_type(key).structural_family}  "
                f"cap_tie={cap_tie_mode_for_layout(key)}  interface={iface}"
            )
        rows = run_cap_peak_ladder(
            key,
            refinements=refinements,
            progress=log,
            require_mesh_qa=require_mesh_qa,
            **kw,
        )
        by_layout[key] = rows
        drifts = summarize_cap_peak_drifts(rows)
        drift_rows.extend(drifts)
        if log is not None:
            cap_notes = [
                d
                for d in drifts
                if d.kind == "cap" and d.metric == "peak_hashin" and d.statistic == "max"
            ]
            note = cap_notes[0].notes if cap_notes else "n/a"
            log(
                f"[cap-peak] [{i}/{len(layout_keys)}] {key}  done in "
                f"{time.perf_counter() - t_layout:.1f}s  verdict={note}  "
                f"elapsed={time.perf_counter() - t0:.1f}s"
            )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_cap_peak_ladder_csv(out / "cap_peak_ladder.csv", by_layout)
        write_cap_peak_locations_json(out / "cap_peak_locations.json", by_layout)
        write_cap_peak_drift_summary_csv(out / "cap_peak_drift_summary.csv", drift_rows)
        if log is not None:
            log(f"[cap-peak] wrote {out / 'cap_peak_ladder.csv'}")
            log(f"[cap-peak] wrote {out / 'cap_peak_locations.json'}")
            log(f"[cap-peak] wrote {out / 'cap_peak_drift_summary.csv'}")
            log(f"[cap-peak] done  layouts={len(layout_keys)}  total={time.perf_counter() - t0:.1f}s")

    return by_layout, drift_rows


__all__ = [
    "DEFAULT_INVESTIGATION_KEYS",
    "ENDPOINT_BAND",
    "METRICS",
    "STATISTICS",
    "CapPeakDriftRow",
    "CapPeakRung",
    "ElementPeakTags",
    "KindMetricStats",
    "PeakLocation",
    "analyze_cap_peak_rung",
    "endpoint_band_mask",
    "percentile_stats",
    "run_cap_peak_ladder",
    "run_cap_peak_study",
    "summarize_cap_peak_drifts",
    "tag_element",
    "verdict_notes",
    "write_cap_peak_drift_summary_csv",
    "write_cap_peak_ladder_csv",
    "write_cap_peak_locations_json",
]
