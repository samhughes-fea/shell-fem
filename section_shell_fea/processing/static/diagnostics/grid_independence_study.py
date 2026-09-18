"""Systematic M+P→S→T→Q mesh-density study across structural system keys.

Sweeps ``target_element_length_frac`` for each layout in
:data:`SYSTEM_TYPE_KEYS` (or a subset). Default load source is Cotide
``new_3.2m_rotor`` + ``NACA_63-415.dat`` (shared station recipe); pass
``load_source='synthetic'`` for the analytic cascade recipe.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TextIO

import numpy as np

from blade_structure_utilities.geometry_utilities.section_geometry.layout.system_layout import (
    SYSTEM_TYPE_KEYS,
    resolve_system_type,
)

# .../section_shell_fea/processing/static/diagnostics/ → repo root is parents[6]
_REPO_ROOT = Path(__file__).resolve().parents[6]

from .asme_gci import GciResult, gci_triplet
from .cascade_mesh_convergence import (
    DEFAULT_REFINEMENTS,
    KIND_PEAK_METRICS,
    KINDS,
    CascadeScalars,
    run_cascade_at_frac,
    series_by_kind,
    write_convergence_csv,
)
from .plot_grid_independence import plot_grid_independence_study

ProgressFn = Callable[[str], None]

STUDY_REFINEMENTS: tuple[float, ...] = (0.024, 0.016, 0.012, 0.008, 0.005, 0.0035)
# Day-to-day CI smoke: must exclude CI_WATCH_KEYS so ``--keys smoke --gates ci`` PASS.
CI_SMOKE_KEYS: tuple[str, ...] = (
    "0A",
    "2A-F",
    "0B",
    "2B-CN",
    "2C-F",
    "5C-CN",
    "2D-CN",
    "2D-F",
)
FAST_CI_KEYS: tuple[str, ...] = ("2A-F", "2D-F")

# Known CI membrane residual set (fine GCI >10% on skin |N12| / mid-σ).
# Full ``--keys all --gates ci`` is expected to fail exactly these until Phase B.
CI_WATCH_KEYS: tuple[str, ...] = (
    "3D-CN",
    "4D-CN",
    "4D-F",
    "5D-CN",
    "5D-F",
)
# Study FI / governing_fi residual set (report-only; not CI).
STUDY_FI_WATCH_KEYS: tuple[str, ...] = (
    "2D-CN",
    "3B-CN",
    "3D-CN",
    "4D-CN",
    "5D-CN",
    "5D-F",
)

# Study targets (CLI ``--gates study``): plateau at production→finer.
MAX_FULL_LADDER_DRIFT = 2.0
STUDY_TARGET_LAST_TWO_DRIFT = 1.25
# CI / default CLI: same full-ladder band as test_cascade_mesh_convergence; last-two
# uses the same 2× band until the finer rung is proven (summary still reports 1.25 target).
CI_MAX_LAST_TWO_DRIFT = 2.0
# ASME / Roache fine-grid GCI on the three finest rungs (fraction of |Q1|).
CI_MAX_GCI_FINE = 0.10
STUDY_MAX_GCI_FINE = 0.05

# Proven CI hard-gates (post junction-local h remediation): membrane + mid-σ.
CI_KIND_METRICS = ("peak_abs_n12", "peak_mid_sig")
# Study mode also tracks Hashin in-plane (D-family ownership still under work).
STUDY_KIND_METRICS = ("peak_abs_n12", "peak_mid_sig", "peak_hashin")
# Back-compat alias used by summary tables / older tests (= study set).
KIND_METRICS = STUDY_KIND_METRICS
# Full P→S→T→Q kind-peak report set (ladder CSV + GCI summary tables).
REPORT_KIND_METRICS = KIND_PEAK_METRICS
PRODUCTION_SUBLADDER = tuple(float(x) for x in DEFAULT_REFINEMENTS)


@dataclass(frozen=True)
class GateFailure:
    """One acceptance-gate violation."""

    layout: str
    kind: str
    metric: str
    drift: float
    limit: float
    detail: str = ""


@dataclass(frozen=True)
class LayoutDriftSummary:
    """Per-layout drift / trust / GCI summary across the density ladder."""

    layout: str
    family: str
    has_spar_caps: bool
    present_kinds: tuple[str, ...]
    full_drift: dict[str, dict[str, float]] = field(default_factory=dict)
    last_two_drift: dict[str, dict[str, float]] = field(default_factory=dict)
    production_drift: dict[str, dict[str, float]] = field(default_factory=dict)
    # Endpoint-excluded peaks (study-mode cap bulk plateau)
    interior_last_two_drift: dict[str, dict[str, float]] = field(default_factory=dict)
    # ASME Roache GCI on three finest rungs (raw peaks)
    gci_fine: dict[str, dict[str, float]] = field(default_factory=dict)
    p_obs: dict[str, dict[str, float]] = field(default_factory=dict)
    q_ext: dict[str, dict[str, float]] = field(default_factory=dict)
    gci_asymptotic_ratio: dict[str, dict[str, float]] = field(default_factory=dict)
    gci_ok: dict[str, dict[str, bool]] = field(default_factory=dict)
    gci_detail: dict[str, dict[str, str]] = field(default_factory=dict)
    # Cap / endpoint-excluded GCI (reported; study/CI policy as for interior drifts)
    interior_gci_fine: dict[str, dict[str, float]] = field(default_factory=dict)
    interior_gci_ok: dict[str, dict[str, bool]] = field(default_factory=dict)
    interior_gci_detail: dict[str, dict[str, str]] = field(default_factory=dict)
    governing_fi_full_drift: float = float("nan")
    governing_fi_last_two_drift: float = float("nan")
    governing_fi_production_drift: float = float("nan")
    governing_fi_gci_fine: float = float("nan")
    governing_fi_p_obs: float = float("nan")
    governing_fi_q_ext: float = float("nan")
    governing_fi_gci_asymptotic_ratio: float = float("nan")
    governing_fi_gci_ok: bool = False
    governing_fi_gci_detail: str = ""
    max_residual_rel: float = float("nan")
    all_mesh_ok: bool = False
    all_trust_ok: bool = False

# Kind peaks that are singularity-dominated under skin_offset (CI informational).
INFORMATIONAL_KINDS: frozenset[str] = frozenset({"cap"})
INTERIOR_METRIC_ATTR = {m: f"{m}_interior" for m in REPORT_KIND_METRICS}


def present_kinds(row: CascadeScalars) -> tuple[str, ...]:
    """Kinds with at least one mesh element on this rung."""
    return tuple(k for k in KINDS if int(row.n_elem.get(k, 0) or 0) > 0)


def last_two_drift(values: Sequence[float]) -> float:
    """Ratio of last rung to penultimate (plateau at production→finer)."""
    if len(values) < 2:
        raise ValueError("last_two_drift requires at least two ladder values")
    a = float(values[-2])
    b = float(values[-1])
    if not (np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0):
        raise AssertionError(f"non-finite or non-positive last-two values: {a}, {b}")
    return max(a, b) / min(a, b)


def _safe_drift(values: Sequence[float]) -> float:
    """Like :func:`drift` but returns NaN when the series is not positive."""
    finite = [float(v) for v in values if np.isfinite(v) and float(v) > 0.0]
    if len(finite) != len(values) or not finite:
        return float("nan")
    return max(finite) / min(finite)


def _safe_last_two(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    try:
        return last_two_drift(values)
    except (AssertionError, ValueError):
        return float("nan")


def _finest_three_h_q(
    rows: Sequence[CascadeScalars],
    values: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Pair ``h_m`` with metric values on the three finest rungs (by ``h_m``)."""
    if len(rows) != len(values):
        raise ValueError("rows and values length mismatch")
    pairs = sorted(
        (
            (float(r.h_m), float(v))
            for r, v in zip(rows, values)
            if np.isfinite(r.h_m) and np.isfinite(v)
        ),
        key=lambda t: t[0],
    )
    if len(pairs) > 3:
        pairs = pairs[:3]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _gci_from_series(
    rows: Sequence[CascadeScalars],
    values: Sequence[float],
) -> GciResult:
    h, q = _finest_three_h_q(rows, values)
    return gci_triplet(h, q)


def cap_tie_mode_for_layout(layout: str) -> str:
    """Production ties on cap-bearing layouts; none for A-family."""
    return "skin_offset" if bool(resolve_system_type(layout).has_spar_caps) else "none"


@dataclass(frozen=True)
class CotideStationRecipe:
    """Shared station geometry + beam resultants for multi-key mesh ladders."""

    rotor_dir: str
    naca_dat: str
    zL: float
    chord_m: float
    twist_deg: float
    thickness: dict[str, float]
    load: dict[str, float]
    station_label: str
    M_flapwise: float
    M_edgewise: float
    V_flapwise: float
    V_edgewise: float
    T: float

    def cascade_kwargs(self) -> dict[str, Any]:
        """Kwargs for :func:`run_cascade_at_frac` / :func:`run_layout_ladder`."""
        return {
            "chord": float(self.chord_m),
            "twist_deg": float(self.twist_deg),
            "thickness": dict(self.thickness),
            "load": dict(self.load),
            "naca_dat": str(self.naca_dat),
        }


def load_cotide_station_recipe(
    *,
    repo_root: Path | None = None,
    rotor_dir: Path | None = None,
    naca_dat: Path | None = None,
    zL: float | None = None,
    grid_n_sections: int = 32,
) -> CotideStationRecipe:
    """Load ``new_3.2m_rotor`` + ``NACA_63-415.dat`` and pick a station recipe.

    Default station is max ``|m_YB|`` on the InternalForce worst-case envelope
    (root for the 3.2 m rotor). Thicknesses are structural floors (not sized).
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    rotor = Path(rotor_dir) if rotor_dir is not None else root / "cotide_data_interfaces" / "new_3.2m_rotor"
    naca = Path(naca_dat) if naca_dat is not None else root / "cotide_data_interfaces" / "NACA_63-415.dat"
    if not rotor.is_dir():
        raise FileNotFoundError(f"rotor dir not found: {rotor}")
    if not naca.is_file():
        raise FileNotFoundError(f"NACA dat not found: {naca}")

    from blade_structure_inputs import BladeStructureInputs
    from blade_structure_precompute.module.internal_force import InternalForce
    from blade_structure_precompute.examples._shared.scaled_geometry import analysis_geometry_from_blade
    from blade_structure_utilities.geometry_utilities.coordinate_frames import (
        blade_resultants_to_shell_channels,
    )

    inputs = BladeStructureInputs(
        rotor_dir=rotor,
        naca_file=naca,
        internal_inputs_dir=root / "blade_structure_inputs",
        structural_system_key="5D-F",
        component_materials={"skin": 2, "spar": 2, "web": 2},
        structural_min_t_skin=0.001,
        structural_min_t_spar=0.002,
        structural_min_t_web=0.0015,
        grid_n_sections=int(grid_n_sections),
    ).load()
    bg = inputs.blade_geometry
    ld = inputs.loading
    scale = analysis_geometry_from_blade(bg, inputs.sampling_distribution_parameters)
    ifu = InternalForce(scaling=scale, loading=ld)
    ifu.execute()
    wc = ifu.get_worst_curves_dict()

    z_grid = np.asarray(wc["m_YB"].zL, dtype=float)
    m_flap = np.asarray(wc["m_YB"].values, dtype=float)
    if zL is None:
        i_star = int(np.argmax(np.abs(m_flap)))
        z_star = float(z_grid[i_star])
        station_label = "max_abs_m_YB"
    else:
        z_star = float(zL)
        station_label = f"zL={z_star:.6g}"

    def _at(name: str) -> float:
        curve = wc[name]
        return float(np.interp(z_star, np.asarray(curve.zL, dtype=float), np.asarray(curve.values, dtype=float)))

    M_flap = _at("m_YB")
    V_flap = _at("v_XB")
    M_edge = _at("m_XB")
    V_edge = _at("v_YB")
    T = _at("t_ZB") if "t_ZB" in wc else 0.0
    ch = blade_resultants_to_shell_channels(
        M_flapwise=M_flap,
        V_flapwise=V_flap,
        M_edgewise=M_edge,
        V_edgewise=V_edge,
        T=T,
    )
    chord_m = float(np.interp(z_star, np.asarray(bg.r_over_R, dtype=float), np.asarray(bg.c, dtype=float)))
    twist_deg = float(
        np.interp(z_star, np.asarray(bg.r_over_R, dtype=float), np.asarray(bg.twist, dtype=float))
    )
    ssp = inputs.structural_system_parameters
    thickness = {
        "skin": float(getattr(ssp, "structural_min_t_skin", 0.001) or 0.001),
        "spar": float(getattr(ssp, "structural_min_t_spar", 0.002) or 0.002),
        "web": float(getattr(ssp, "structural_min_t_web", 0.0015) or 0.0015),
    }
    load = {
        "N": float(ch.N),
        "Vy": float(ch.Vy),
        "Vz": float(ch.Vz),
        "My": float(ch.My),
        "Mz": float(ch.Mz),
        "T": float(ch.T),
    }
    return CotideStationRecipe(
        rotor_dir=str(rotor.resolve()),
        naca_dat=str(naca.resolve()),
        zL=z_star,
        chord_m=chord_m,
        twist_deg=twist_deg,
        thickness=thickness,
        load=load,
        station_label=station_label,
        M_flapwise=M_flap,
        M_edgewise=M_edge,
        V_flapwise=V_flap,
        V_edgewise=V_edge,
        T=float(T),
    )


def _default_progress(stream: TextIO = sys.stdout) -> ProgressFn:
    def _emit(msg: str) -> None:
        print(msg, file=stream, flush=True)

    return _emit


def run_layout_ladder(
    layout: str,
    *,
    refinements: Sequence[float] = STUDY_REFINEMENTS,
    interface_constraint_mode: str = "shared",
    require_mesh_qa: bool = True,
    progress: ProgressFn | None = None,
    backend: str = "mitc4",
    work_dir: str | Path | None = None,
    ccx_exe: str = "ccx",
    **kwargs: Any,
) -> list[CascadeScalars]:
    """Density ladder for one structural system key.

    ``backend``:
    - ``mitc4`` (default): native ``run_cascade_at_frac``
    - ``ccx``: live CalculiX membrane cascade (``run_ccx_cascade_at_frac``)
    """
    be = str(backend or "mitc4").lower().strip()
    if be not in ("mitc4", "ccx"):
        raise ValueError(f"backend must be 'mitc4' or 'ccx', got {backend!r}")
    tie = cap_tie_mode_for_layout(layout)
    n_rung = len(tuple(refinements))
    rows: list[CascadeScalars] = []
    for ri, frac in enumerate(refinements, start=1):
        t0 = time.perf_counter()
        if progress is not None:
            progress(
                f"    rung {ri}/{n_rung}  frac={float(frac):.4f}  "
                f"backend={be}  cap_tie={tie}  interface={interface_constraint_mode}  ..."
            )
        if be == "ccx":
            from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.ccx_cascade_mesh_convergence import (
                run_ccx_cascade_at_frac,
            )

            row = run_ccx_cascade_at_frac(
                float(frac),
                layout=layout,
                cap_tie_mode=tie,
                interface_constraint_mode=interface_constraint_mode,
                require_mesh_qa=require_mesh_qa,
                work_dir=work_dir,
                ccx_exe=ccx_exe,
                **kwargs,
            )
        else:
            row = run_cascade_at_frac(
                float(frac),
                layout=layout,
                cap_tie_mode=tie,
                interface_constraint_mode=interface_constraint_mode,
                require_mesh_qa=require_mesh_qa,
                **kwargs,
            )
        rows.append(row)
        if progress is not None:
            n_tot = sum(int(row.n_elem.get(k, 0) or 0) for k in KINDS)
            kinds = ",".join(present_kinds(row)) or "-"
            peak_n12 = row.peak_abs_n12.get("skin", float("nan"))
            progress(
                f"    rung {ri}/{n_rung}  done in {time.perf_counter() - t0:.1f}s  "
                f"n_elem={n_tot}  kinds=[{kinds}]  "
                f"mesh_ok={row.mesh_ok}  trust={row.cascade_trust_ok}  "
                f"peak_|N12|_skin={peak_n12:.4g}  "
                f"gov_fi={row.governing_fi:.4g}"
            )
    return rows


def _production_rows(rows: Sequence[CascadeScalars]) -> list[CascadeScalars]:
    """Rungs at the historic 3-level production ladder fracs (when present)."""
    wanted = set(PRODUCTION_SUBLADDER)
    out = [r for r in rows if float(r.frac) in wanted]
    return out if len(out) >= 2 else list(rows[:3]) if len(rows) >= 2 else list(rows)


def summarize_layout_ladder(rows: Sequence[CascadeScalars]) -> LayoutDriftSummary:
    """Compute drifts and ASME/Roache GCI on the three finest rungs."""
    if not rows:
        raise ValueError("rows is empty")
    layout = str(rows[0].layout or "")
    spec = resolve_system_type(layout)
    kinds = present_kinds(rows[-1])
    kinds = tuple(
        k for k in kinds if all(int(r.n_elem.get(k, 0) or 0) > 0 for r in rows)
    )

    full: dict[str, dict[str, float]] = {m: {} for m in REPORT_KIND_METRICS}
    last2: dict[str, dict[str, float]] = {m: {} for m in REPORT_KIND_METRICS}
    prod: dict[str, dict[str, float]] = {m: {} for m in REPORT_KIND_METRICS}
    interior_lt: dict[str, dict[str, float]] = {m: {} for m in REPORT_KIND_METRICS}
    gci_f: dict[str, dict[str, float]] = {m: {} for m in REPORT_KIND_METRICS}
    p_obs: dict[str, dict[str, float]] = {m: {} for m in REPORT_KIND_METRICS}
    q_ext: dict[str, dict[str, float]] = {m: {} for m in REPORT_KIND_METRICS}
    gci_asym: dict[str, dict[str, float]] = {m: {} for m in REPORT_KIND_METRICS}
    gci_ok: dict[str, dict[str, bool]] = {m: {} for m in REPORT_KIND_METRICS}
    gci_det: dict[str, dict[str, str]] = {m: {} for m in REPORT_KIND_METRICS}
    int_gci_f: dict[str, dict[str, float]] = {m: {} for m in REPORT_KIND_METRICS}
    int_gci_ok: dict[str, dict[str, bool]] = {m: {} for m in REPORT_KIND_METRICS}
    int_gci_det: dict[str, dict[str, str]] = {m: {} for m in REPORT_KIND_METRICS}
    prod_rows = _production_rows(rows)
    for metric in REPORT_KIND_METRICS:
        series = series_by_kind(rows, metric)
        series_p = series_by_kind(prod_rows, metric)
        series_int = series_by_kind(rows, INTERIOR_METRIC_ATTR[metric])
        for kind in kinds:
            full[metric][kind] = _safe_drift(series[kind])
            last2[metric][kind] = _safe_last_two(series[kind])
            prod[metric][kind] = _safe_drift(series_p[kind])
            interior_lt[metric][kind] = _safe_last_two(series_int[kind])
            gci = _gci_from_series(rows, series[kind])
            gci_f[metric][kind] = float(gci.gci_fine)
            p_obs[metric][kind] = float(gci.p_obs)
            q_ext[metric][kind] = float(gci.q_ext)
            gci_asym[metric][kind] = float(gci.asymptotic_ratio)
            gci_ok[metric][kind] = bool(gci.ok)
            gci_det[metric][kind] = str(gci.detail)
            gci_i = _gci_from_series(rows, series_int[kind])
            int_gci_f[metric][kind] = float(gci_i.gci_fine)
            int_gci_ok[metric][kind] = bool(gci_i.ok)
            int_gci_det[metric][kind] = str(gci_i.detail)

    gov = [float(r.governing_fi) for r in rows]
    gov_p = [float(r.governing_fi) for r in prod_rows]
    gov_gci = _gci_from_series(rows, gov)
    residuals = [float(r.residual_rel) for r in rows if np.isfinite(r.residual_rel)]
    return LayoutDriftSummary(
        layout=layout,
        family=str(spec.structural_family),
        has_spar_caps=bool(spec.has_spar_caps),
        present_kinds=kinds,
        full_drift=full,
        last_two_drift=last2,
        production_drift=prod,
        interior_last_two_drift=interior_lt,
        gci_fine=gci_f,
        p_obs=p_obs,
        q_ext=q_ext,
        gci_asymptotic_ratio=gci_asym,
        gci_ok=gci_ok,
        gci_detail=gci_det,
        interior_gci_fine=int_gci_f,
        interior_gci_ok=int_gci_ok,
        interior_gci_detail=int_gci_det,
        governing_fi_full_drift=_safe_drift(gov),
        governing_fi_last_two_drift=_safe_last_two(gov),
        governing_fi_production_drift=_safe_drift(gov_p),
        governing_fi_gci_fine=float(gov_gci.gci_fine),
        governing_fi_p_obs=float(gov_gci.p_obs),
        governing_fi_q_ext=float(gov_gci.q_ext),
        governing_fi_gci_asymptotic_ratio=float(gov_gci.asymptotic_ratio),
        governing_fi_gci_ok=bool(gov_gci.ok),
        governing_fi_gci_detail=str(gov_gci.detail),
        max_residual_rel=float(max(residuals)) if residuals else float("nan"),
        all_mesh_ok=all(bool(r.mesh_ok) for r in rows),
        all_trust_ok=all(bool(r.cascade_trust_ok) for r in rows),
    )


def _append_gci_gate(
    fails: list[GateFailure],
    *,
    layout: str,
    kind: str,
    metric: str,
    gci: float,
    ok: bool,
    detail: str,
    limit: float,
) -> None:
    """Fail on fine-grid GCI above ``limit``, or on a GCI that is undefined.

    Classification is delegated to :func:`classify_gci_fail` rather than decided
    here, so the gate and the stability scorecard cannot disagree about what
    counts as a GCI failure.

    A non-monotonic triple is **not** a failure on its own. Roache's GCI assumes
    monotonic convergence, so when the fine-grid GCI is far below the limit the
    triple is noise and the extrapolation simply does not apply — the classifier
    calls that ``roache_applicability``. This gate previously failed on
    ``gci_non_monotonic`` unconditionally, which rejected cells that were
    converged to `2e-4` against a limit of `0.1`.
    """
    # Local import avoids circular import with stability_scorecard (which imports
    # CI_* constants from this module).
    from .stability_scorecard import classify_gci_fail

    classification = classify_gci_fail(
        gci_fine=float(gci),
        gci_ok=bool(ok),
        detail=str(detail or ""),
        limit=float(limit),
    )
    if classification is None or classification == "roache_applicability":
        return
    fails.append(
        GateFailure(
            layout,
            kind,
            f"{metric}_gci_fine",
            float(gci) if np.isfinite(gci) else float("nan"),
            float(limit),
            "asme_roache_gci" if classification == "magnitude" else "gci_undefined",
        )
    )


def evaluate_layout_gates(
    summary: LayoutDriftSummary,
    *,
    gates: str = "ci",
    max_full_drift: float = MAX_FULL_LADDER_DRIFT,
    max_last_two: float | None = None,
    max_gci_fine: float | None = None,
    membrane_only: bool = False,
) -> list[GateFailure]:
    """Return gate failures for mesh/trust, S/T/Q drift, plateau, and GCI.

    ``gates="ci"`` (default): production-subladder full drift ≤ ``max_full_drift``,
    last-two ≤ ``CI_MAX_LAST_TWO_DRIFT``, and fine-grid GCI ≤ ``CI_MAX_GCI_FINE``
    on skin/web ``peak_abs_n12`` + ``peak_mid_sig`` only (proven freeze). Hashin /
    ``governing_fi`` stay study-reported. Cap kind-peaks are informational under
    ``skin_offset``.

    ``gates="study"``: skin/web use full ladder ≤ ``max_full_drift`` and last-two
    ≤ ``STUDY_TARGET_LAST_TWO_DRIFT``; fine-grid GCI ≤ ``STUDY_MAX_GCI_FINE`` on
    ``peak_abs_n12`` / ``peak_mid_sig`` / ``peak_hashin`` + ``governing_fi``; cap
    uses endpoint-excluded last-two only ≤ ``CI_MAX_LAST_TWO_DRIFT`` (interior GCI
    reported, not gated).

    ``gates="ccx"``: membrane-only CI profile — skin/web ``peak_abs_n12`` drift +
    GCI only (Hashin / mid_sig / governing_fi skipped). Cap ``peak_abs_n12`` is
    informational. Equivalent to ``membrane_only=True`` with CI numeric limits.
    CCX peak ``|N11|``/``|N22|`` remain report-only.

    When ``membrane_only=True`` (also set by ``gates="ccx"``, or CCX +
    ``gates="study"``), only ``peak_abs_n12`` is gated and governing_fi is skipped.
    """
    mode = str(gates).lower().strip()
    if mode == "ccx":
        membrane_only = True
        limit_mode = "ci"
    else:
        limit_mode = mode
    if limit_mode not in ("ci", "study"):
        raise ValueError(f"gates must be 'ci', 'study', or 'ccx'; got {gates!r}")

    if max_last_two is None:
        max_last_two = (
            STUDY_TARGET_LAST_TWO_DRIFT
            if limit_mode == "study"
            else CI_MAX_LAST_TWO_DRIFT
        )
    if max_gci_fine is None:
        max_gci_fine = (
            STUDY_MAX_GCI_FINE if limit_mode == "study" else CI_MAX_GCI_FINE
        )
    use_production = limit_mode != "study"
    drift_table = summary.production_drift if use_production else summary.full_drift
    gov_full = (
        summary.governing_fi_production_drift
        if use_production
        else summary.governing_fi_full_drift
    )
    full_suffix = "production" if use_production else "full"
    if membrane_only:
        kind_metrics: tuple[str, ...] = ("peak_abs_n12",)
        gci_metrics: tuple[str, ...] = ("peak_abs_n12",)
    elif limit_mode == "ci":
        kind_metrics = CI_KIND_METRICS
        gci_metrics = CI_KIND_METRICS
    else:
        kind_metrics = STUDY_KIND_METRICS
        gci_metrics = STUDY_KIND_METRICS

    fails: list[GateFailure] = []
    if not summary.all_mesh_ok:
        fails.append(
            GateFailure(
                summary.layout, "-", "mesh_ok", float("nan"), 1.0, "mesh QA failed on a rung"
            )
        )
    if not summary.all_trust_ok:
        fails.append(
            GateFailure(
                summary.layout,
                "-",
                "cascade_trust_ok",
                float("nan"),
                1.0,
                "cascade trust failed on a rung",
            )
        )

    for metric in kind_metrics:
        for kind in summary.present_kinds:
            if kind in INFORMATIONAL_KINDS:
                if limit_mode != "study":
                    continue
                # Study: cap bulk plateau via interior last-two only (limit 2.0).
                d_lt = summary.interior_last_two_drift.get(metric, {}).get(
                    kind, float("nan")
                )
                if np.isfinite(d_lt) and d_lt > CI_MAX_LAST_TWO_DRIFT:
                    fails.append(
                        GateFailure(
                            summary.layout,
                            kind,
                            f"{metric}_interior_last_two",
                            float(d_lt),
                            CI_MAX_LAST_TWO_DRIFT,
                            "study-mode cap bulk plateau (endpoint-excluded)",
                        )
                    )
                continue

            d_full = drift_table.get(metric, {}).get(kind, float("nan"))
            if np.isfinite(d_full) and d_full > max_full_drift:
                fails.append(
                    GateFailure(
                        summary.layout,
                        kind,
                        f"{metric}_{full_suffix}",
                        float(d_full),
                        max_full_drift,
                    )
                )
            d_lt = summary.last_two_drift.get(metric, {}).get(kind, float("nan"))
            if np.isfinite(d_lt) and d_lt > max_last_two:
                fails.append(
                    GateFailure(
                        summary.layout,
                        kind,
                        f"{metric}_last_two",
                        float(d_lt),
                        float(max_last_two),
                    )
                )
            if metric in gci_metrics:
                _append_gci_gate(
                    fails,
                    layout=summary.layout,
                    kind=kind,
                    metric=metric,
                    gci=summary.gci_fine.get(metric, {}).get(kind, float("nan")),
                    ok=bool(summary.gci_ok.get(metric, {}).get(kind, False)),
                    detail=str(summary.gci_detail.get(metric, {}).get(kind, "")),
                    limit=float(max_gci_fine),
                )

    if membrane_only or limit_mode != "study":
        # CI freeze: Hashin / governing_fi are study-reported only (CSV + study gates).
        return fails

    # governing FI: study mode only (full ladder + last-two + fine GCI).
    if np.isfinite(gov_full) and gov_full > max_full_drift:
        fails.append(
            GateFailure(
                summary.layout,
                "-",
                f"governing_fi_{full_suffix}",
                float(gov_full),
                max_full_drift,
            )
        )
    g_lt = summary.governing_fi_last_two_drift
    if np.isfinite(g_lt) and g_lt > max_last_two:
        fails.append(
            GateFailure(
                summary.layout,
                "-",
                "governing_fi_last_two",
                float(g_lt),
                float(max_last_two),
            )
        )
    _append_gci_gate(
        fails,
        layout=summary.layout,
        kind="-",
        metric="governing_fi",
        gci=summary.governing_fi_gci_fine,
        ok=bool(summary.governing_fi_gci_ok),
        detail=str(summary.governing_fi_gci_detail),
        limit=float(max_gci_fine),
    )
    return fails


def write_summary_csv(
    path: str | Path,
    summaries: Iterable[LayoutDriftSummary],
) -> Path:
    """One row per layout with drift / trust columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layout",
        "family",
        "has_spar_caps",
        "present_kinds",
        "all_mesh_ok",
        "all_trust_ok",
        "max_residual_rel",
        "governing_fi_full_drift",
        "governing_fi_production_drift",
        "governing_fi_last_two_drift",
        "governing_fi_gci_fine",
        "governing_fi_p_obs",
        "governing_fi_q_ext",
        "governing_fi_gci_asymptotic_ratio",
        "governing_fi_gci_ok",
        "governing_fi_gci_detail",
        "study_target_last_two",
        "ci_max_gci_fine",
        "study_max_gci_fine",
    ]
    for metric in REPORT_KIND_METRICS:
        for kind in KINDS:
            fieldnames.append(f"{metric}_full_{kind}")
            fieldnames.append(f"{metric}_production_{kind}")
            fieldnames.append(f"{metric}_last_two_{kind}")
            fieldnames.append(f"{metric}_interior_last_two_{kind}")
            fieldnames.append(f"{metric}_gci_fine_{kind}")
            fieldnames.append(f"{metric}_p_obs_{kind}")
            fieldnames.append(f"{metric}_q_ext_{kind}")
            fieldnames.append(f"{metric}_gci_asym_{kind}")
            fieldnames.append(f"{metric}_gci_ok_{kind}")
            fieldnames.append(f"{metric}_gci_detail_{kind}")
            fieldnames.append(f"{metric}_interior_gci_fine_{kind}")

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            rec: dict[str, Any] = {
                "layout": s.layout,
                "family": s.family,
                "has_spar_caps": s.has_spar_caps,
                "present_kinds": "|".join(s.present_kinds),
                "all_mesh_ok": s.all_mesh_ok,
                "all_trust_ok": s.all_trust_ok,
                "max_residual_rel": s.max_residual_rel,
                "governing_fi_full_drift": s.governing_fi_full_drift,
                "governing_fi_production_drift": s.governing_fi_production_drift,
                "governing_fi_last_two_drift": s.governing_fi_last_two_drift,
                "governing_fi_gci_fine": s.governing_fi_gci_fine,
                "governing_fi_p_obs": s.governing_fi_p_obs,
                "governing_fi_q_ext": s.governing_fi_q_ext,
                "governing_fi_gci_asymptotic_ratio": s.governing_fi_gci_asymptotic_ratio,
                "governing_fi_gci_ok": s.governing_fi_gci_ok,
                "governing_fi_gci_detail": s.governing_fi_gci_detail,
                "study_target_last_two": STUDY_TARGET_LAST_TWO_DRIFT,
                "ci_max_gci_fine": CI_MAX_GCI_FINE,
                "study_max_gci_fine": STUDY_MAX_GCI_FINE,
            }
            for metric in REPORT_KIND_METRICS:
                for kind in KINDS:
                    rec[f"{metric}_full_{kind}"] = s.full_drift.get(metric, {}).get(
                        kind, float("nan")
                    )
                    rec[f"{metric}_production_{kind}"] = s.production_drift.get(
                        metric, {}
                    ).get(kind, float("nan"))
                    rec[f"{metric}_last_two_{kind}"] = s.last_two_drift.get(
                        metric, {}
                    ).get(kind, float("nan"))
                    rec[f"{metric}_interior_last_two_{kind}"] = (
                        s.interior_last_two_drift.get(metric, {}).get(kind, float("nan"))
                    )
                    rec[f"{metric}_gci_fine_{kind}"] = s.gci_fine.get(metric, {}).get(
                        kind, float("nan")
                    )
                    rec[f"{metric}_p_obs_{kind}"] = s.p_obs.get(metric, {}).get(
                        kind, float("nan")
                    )
                    rec[f"{metric}_q_ext_{kind}"] = s.q_ext.get(metric, {}).get(
                        kind, float("nan")
                    )
                    rec[f"{metric}_gci_asym_{kind}"] = s.gci_asymptotic_ratio.get(
                        metric, {}
                    ).get(kind, float("nan"))
                    rec[f"{metric}_gci_ok_{kind}"] = s.gci_ok.get(metric, {}).get(
                        kind, False
                    )
                    rec[f"{metric}_gci_detail_{kind}"] = s.gci_detail.get(metric, {}).get(
                        kind, ""
                    )
                    rec[f"{metric}_interior_gci_fine_{kind}"] = s.interior_gci_fine.get(
                        metric, {}
                    ).get(kind, float("nan"))
            writer.writerow(rec)
    return path


def resolve_study_keys(spec: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse ``all`` / ``smoke`` / ``watch`` / ``fast`` / comma-list into keys."""
    if spec is None or spec == "all":
        return tuple(SYSTEM_TYPE_KEYS)
    if isinstance(spec, str):
        text = spec.strip()
        if text.lower() == "all":
            return tuple(SYSTEM_TYPE_KEYS)
        if text.lower() == "smoke":
            return CI_SMOKE_KEYS
        if text.lower() == "watch":
            return CI_WATCH_KEYS
        if text.lower() == "fast":
            return FAST_CI_KEYS
        keys = tuple(k.strip() for k in text.split(",") if k.strip())
    else:
        keys = tuple(str(k) for k in spec)
    unknown = [k for k in keys if k not in SYSTEM_TYPE_KEYS]
    if unknown:
        raise ValueError(f"unknown layout key(s): {unknown}")
    return keys


def run_grid_independence_study(
    keys: str | Sequence[str] | None = "all",
    *,
    refinements: Sequence[float] = STUDY_REFINEMENTS,
    out_dir: str | Path | None = None,
    require_mesh_qa: bool = True,
    gates: str = "ci",
    progress: bool | ProgressFn = False,
    load_source: str = "cotide",
    recipe: CotideStationRecipe | None = None,
    zL: float | None = None,
    backend: str = "mitc4",
    ccx_exe: str = "ccx",
    interface_constraint_mode: str = "shared",
    junction_band_elements: int | None = None,
    junction_band_step_over_h: float | None = None,
    junction_band_panel_labels: Sequence[str] | None = None,
    junction_band_which: str = "both",
) -> tuple[dict[str, list[CascadeScalars]], list[LayoutDriftSummary], list[GateFailure]]:
    """Run ladders for ``keys``; optionally write ladder + summary CSVs.

    ``progress=True`` prints detailed per-layout / per-rung status to stdout.
    Pass a callable ``(msg: str) -> None`` for a custom sink.

    ``load_source``:
    - ``cotide`` (default): ``new_3.2m_rotor`` + ``NACA_63-415.dat``, station at
      max ``|m_YB|`` (or ``zL=``), structural-floor thicknesses.
    - ``synthetic``: hard-coded cascade defaults (analytic NACA series).

    ``backend``:
    - ``mitc4`` (default): native FSDT cascade
    - ``ccx``: live CalculiX membrane cascade (gates default to membrane-only)

    ``interface_constraint_mode``: ``shared`` (production), ``transformed_basis`` /
    ``shared_rotated`` (Defect K), or study-only ``hybrid_collinear`` /
    ``hybrid_kind_crossing``.
    """
    layout_keys = resolve_study_keys(keys)
    be = str(backend or "mitc4").lower().strip()
    if be not in ("mitc4", "ccx"):
        raise ValueError(f"backend must be 'mitc4' or 'ccx', got {backend!r}")
    iface = str(interface_constraint_mode or "shared").strip()
    _allowed_iface = (
        "shared",
        "transformed_basis",
        "shared_rotated",
        "hybrid_collinear",
        "hybrid_kind_crossing",
        "shared_cluster_frame",
    )
    if iface not in _allowed_iface:
        raise ValueError(
            f"interface_constraint_mode must be one of {_allowed_iface}, got {iface!r}"
        )
    log: ProgressFn | None
    if progress is True:
        log = _default_progress()
    elif progress is False or progress is None:
        log = None
    else:
        log = progress

    source = str(load_source).lower().strip()
    cascade_kw: dict[str, Any] = {}
    active_recipe: CotideStationRecipe | None = None
    if source == "cotide":
        active_recipe = recipe or load_cotide_station_recipe(zL=zL)
        cascade_kw = active_recipe.cascade_kwargs()
    elif source == "synthetic":
        cascade_kw = {}
    else:
        raise ValueError(f"load_source must be 'cotide' or 'synthetic'; got {load_source!r}")

    if junction_band_elements is not None:
        cascade_kw["junction_band_elements"] = int(junction_band_elements)
    if junction_band_step_over_h is not None:
        cascade_kw["junction_band_step_over_h"] = float(junction_band_step_over_h)
    if junction_band_panel_labels is not None:
        cascade_kw["junction_band_panel_labels"] = tuple(
            str(x) for x in junction_band_panel_labels
        )
    if junction_band_which and str(junction_band_which) != "both":
        cascade_kw["junction_band_which"] = str(junction_band_which)

    gate_mode = str(gates).lower().strip()
    if be == "ccx" and gate_mode == "ci":
        gate_mode = "ccx"
    membrane_only = be == "ccx" or gate_mode == "ccx"

    n_keys = len(layout_keys)
    n_rung = len(tuple(refinements))
    t_study = time.perf_counter()
    ccx_work = Path(out_dir) / "ccx_work" if out_dir is not None else None
    if log is not None:
        jb_n = cascade_kw.get("junction_band_elements", "default")
        jb_step = cascade_kw.get("junction_band_step_over_h", "default")
        jb_labs = cascade_kw.get("junction_band_panel_labels", "all")
        jb_which = cascade_kw.get("junction_band_which", "both")
        log(
            f"[grid-independence] start  layouts={n_keys}  "
            f"rungs={n_rung}  fracs={list(map(float, refinements))}  "
            f"backend={be}  gates={gate_mode}  load_source={source}  "
            f"interface={iface}  junction_band={jb_n}/{jb_step}  "
            f"labels={jb_labs}  which={jb_which}"
        )
        if active_recipe is not None:
            log(
                f"[grid-independence] cotide  rotor={active_recipe.rotor_dir}  "
                f"naca={active_recipe.naca_dat}"
            )
            log(
                f"[grid-independence] station  {active_recipe.station_label}  "
                f"zL={active_recipe.zL:.6g}  chord={active_recipe.chord_m:.6g} m  "
                f"twist={active_recipe.twist_deg:.4g} deg  "
                f"t_skin/spar/web="
                f"{active_recipe.thickness['skin']:.4g}/"
                f"{active_recipe.thickness['spar']:.4g}/"
                f"{active_recipe.thickness['web']:.4g} m"
            )
            log(
                f"[grid-independence] loads  "
                f"Mz(flap)={active_recipe.load['Mz']:.4g}  "
                f"My(edge)={active_recipe.load['My']:.4g}  "
                f"Vz={active_recipe.load['Vz']:.4g}  "
                f"Vy={active_recipe.load['Vy']:.4g}  "
                f"T={active_recipe.load['T']:.4g}"
            )

    by_layout: dict[str, list[CascadeScalars]] = {}
    summaries: list[LayoutDriftSummary] = []
    failures: list[GateFailure] = []
    all_rows: list[CascadeScalars] = []

    for i, key in enumerate(layout_keys, start=1):
        spec = resolve_system_type(key)
        t_layout = time.perf_counter()
        if log is not None:
            log(
                f"[grid-independence] [{i}/{n_keys}] {key}  "
                f"family={spec.structural_family}  "
                f"has_spar_caps={bool(spec.has_spar_caps)}  "
                f"cap_tie={cap_tie_mode_for_layout(key)}"
            )
        rows = run_layout_ladder(
            key,
            refinements=refinements,
            interface_constraint_mode=iface,
            require_mesh_qa=require_mesh_qa,
            progress=log,
            backend=be,
            work_dir=ccx_work,
            ccx_exe=ccx_exe,
            **cascade_kw,
        )
        by_layout[key] = rows
        all_rows.extend(rows)
        summary = summarize_layout_ladder(rows)
        summaries.append(summary)
        layout_fails = evaluate_layout_gates(
            summary, gates=gate_mode, membrane_only=membrane_only
        )
        failures.extend(layout_fails)
        if log is not None:
            gov = summary.governing_fi_last_two_drift
            gov_s = f"{gov:.3f}x" if np.isfinite(gov) else "n/a"
            status = "FAIL" if layout_fails else "ok"
            log(
                f"[grid-independence] [{i}/{n_keys}] {key}  {status}  "
                f"in {time.perf_counter() - t_layout:.1f}s  "
                f"mesh={summary.all_mesh_ok}  trust={summary.all_trust_ok}  "
                f"kinds=[{','.join(summary.present_kinds)}]  "
                f"gov_last_two={gov_s}  "
                f"gate_fails={len(layout_fails)}  "
                f"elapsed={time.perf_counter() - t_study:.1f}s"
            )
            for f in layout_fails:
                d = f"{f.drift:.3f}" if np.isfinite(f.drift) else "nan"
                log(
                    f"      gate  {f.kind}/{f.metric}  drift={d}  limit={f.limit}"
                    + (f"  ({f.detail})" if f.detail else "")
                )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ladder_path = out / "gci_ladder.csv"
        summary_path = out / "gci_summary.csv"
        write_convergence_csv(ladder_path, all_rows)
        write_summary_csv(summary_path, summaries)
        if active_recipe is not None:
            recipe_path = out / "station_recipe.json"
            recipe_path.write_text(
                json.dumps(asdict(active_recipe), indent=2) + "\n", encoding="utf-8"
            )
            if log is not None:
                log(f"[grid-independence] wrote {recipe_path}")
        if log is not None:
            log(f"[grid-independence] wrote {ladder_path}")
            log(f"[grid-independence] wrote {summary_path}")
        try:
            plot_paths = plot_grid_independence_study(
                out,
                backend_label=be.upper(),
                write_hashin=(be == "mitc4"),
            )
            if log is not None:
                for pp in plot_paths:
                    log(f"[grid-independence] wrote {pp}")
        except Exception as exc:  # noqa: BLE001 — plotting must not abort the study
            if log is not None:
                log(f"[grid-independence] plot warning: {exc}")

    if log is not None:
        log(
            f"[grid-independence] done  layouts={n_keys}  "
            f"failures={len(failures)}  "
            f"total={time.perf_counter() - t_study:.1f}s"
        )

    return by_layout, summaries, failures


__all__ = [
    "CI_MAX_GCI_FINE",
    "CI_MAX_LAST_TWO_DRIFT",
    "CI_SMOKE_KEYS",
    "CI_WATCH_KEYS",
    "CotideStationRecipe",
    "FAST_CI_KEYS",
    "GateFailure",
    "CI_KIND_METRICS",
    "INFORMATIONAL_KINDS",
    "KIND_METRICS",
    "REPORT_KIND_METRICS",
    "STUDY_FI_WATCH_KEYS",
    "STUDY_KIND_METRICS",
    "LayoutDriftSummary",
    "MAX_FULL_LADDER_DRIFT",
    "STUDY_MAX_GCI_FINE",
    "STUDY_REFINEMENTS",
    "STUDY_TARGET_LAST_TWO_DRIFT",
    "cap_tie_mode_for_layout",
    "evaluate_layout_gates",
    "last_two_drift",
    "load_cotide_station_recipe",
    "plot_grid_independence_study",
    "present_kinds",
    "resolve_study_keys",
    "run_grid_independence_study",
    "run_layout_ladder",
    "summarize_layout_ladder",
    "write_summary_csv",
]
