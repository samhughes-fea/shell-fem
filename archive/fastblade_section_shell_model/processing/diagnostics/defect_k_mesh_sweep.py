"""Defect K mesh-sweep: secondary traction residuals vs interface mode / mesh density.

Restores the ``SHELL_MODE_SWEEP=1`` workflow documented in ``docs/SHELL_MODEL.md``.
Production periodic FI stays on ``shared`` (Spike 1B parked ``transformed_basis`` K6);
this harness is diagnostic only and must not be wired into optimiser constraints.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from blade_precompute.section_shell_model.engine.global_shell_assembly import (
    assemble_for_periodic_k6,
)
from blade_precompute.section_shell_model.engine.homogenisation import (
    compute_elastic_centroid_from_panels,
    compute_k6_periodic_bc,
)
from blade_precompute.section_shell_model.pre_processing.constitutive.panel_abd import (
    panel_abd_tuple_from_lam,
)
from blade_precompute.section_shell_model.pre_processing.containers.fem_config import (
    MITC4_DEFAULTS,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import (
    ShellPanelResultants,
    ShellTheory,
)
from blade_precompute.section_shell_model.pre_processing.element_library.kernel import (
    resolve_kernel,
)
from blade_precompute.section_shell_model.processing.diagnostics.equilibrium_checks import (
    check_cluster_equilibrium,
    check_panel_equilibrium,
)

_NDOF = MITC4_DEFAULTS.n_dof_node


def _zeros(panels: list[Any]) -> list[np.ndarray]:
    return [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels]


def _panel_edge_element_dofs(
    pm: Any,
    d_global: np.ndarray,
    *,
    edge: str,
    ndof: int = _NDOF,
) -> tuple[np.ndarray, float]:
    """Return 4-node elemental DOFs and L_s for the strip element at panel ``edge``."""
    n = len(pm.s_nodes)
    if n < 2:
        raise ValueError("panel map needs ≥2 contour nodes")
    # Element local layout: bottom i,j then top i,j — strip elements connect s[i]..s[i+1].
    ei = 0 if edge == "start" else n - 2
    L_s = float(abs(pm.s_nodes[ei + 1] - pm.s_nodes[ei]))
    # Local node order for MITC4 strip: (bot_i, bot_j, top_j, top_i) is typical for
    # contour strips; match assembly: bottom row then top row of s_nodes indices.
    bot_i, bot_j = ei, ei + 1
    top_i, top_j = ei + n, ei + 1 + n
    gids = [pm.global_nodes[k] for k in (bot_i, bot_j, top_j, top_i)]
    d_elem = np.zeros(4 * ndof, dtype=float)
    for a, gn in enumerate(gids):
        if gn < 0:
            continue
        d_elem[a * ndof : (a + 1) * ndof] = d_global[gn * ndof : (gn + 1) * ndof]
    return d_elem, L_s


def build_interface_edge_diagnostics(
    panels: list[Any],
    panel_maps: list[Any],
    d_global: np.ndarray,
    *,
    L_x: float = 1.0,
    ndof: int = _NDOF,
) -> list[dict]:
    """Populate ``interface_edge_set`` diagnostics from edge-integrated tractions."""
    kernel = resolve_kernel(ShellTheory.FSDT)
    diags: list[dict] = []
    for pi, (panel, pm) in enumerate(zip(panels, panel_maps)):
        if len(pm.s_nodes) < 2:
            diags.append({})
            continue
        abd, *_rest = panel_abd_tuple_from_lam(panel.lam)
        edge_set: dict[str, dict[str, float]] = {}
        for edge in ("start", "end"):
            d_elem, L_s = _panel_edge_element_dofs(pm, d_global, edge=edge, ndof=ndof)
            out = kernel.recover_edge_shear_traction(
                d_elem, L_s, L_x, abd, edge=edge,
            )
            edge_set[edge] = {
                "Tx_int": float(out["Tx_edge_int"]),
                "Ts_int": float(out["Ts_edge_int"]),
            }
        diags.append({"interface_edge_set": edge_set, "panel_index": pi})
    return diags


def secondary_residuals_for_mode(
    panels: list[Any],
    *,
    interface_constraint_mode: str,
    n_elements_per_panel: int,
    Vy: float = 6.0e3,
    Vz: float = 2.0e3,
    L_x: float = 1.0,
) -> dict[str, float]:
    """Run periodic K6 under (Vy, Vz) and return max secondary traction residuals."""
    z = _zeros(panels)
    ye, ze = compute_elastic_centroid_from_panels(panels)
    K, _f, nm, pm, *_rest = assemble_for_periodic_k6(
        panels,
        z,
        z,
        n_elements_per_panel=n_elements_per_panel,
        interface_constraint_mode=interface_constraint_mode,
    )
    K6, u_modes = compute_k6_periodic_bc(K, nm, pm, panels, ye, ze, L_x=L_x)
    # f_beam = [N, My, Mz, T, Vz, Vy]
    f_beam = np.array([0.0, 0.0, 0.0, 0.0, Vz, Vy], dtype=float)
    try:
        q = np.linalg.solve(K6, f_beam)
    except np.linalg.LinAlgError:
        q = np.linalg.lstsq(K6, f_beam, rcond=None)[0]
    d = u_modes @ q
    diags = build_interface_edge_diagnostics(panels, pm, d, L_x=L_x)
    # Pair-wise check requires non-empty resultant nests (values unused when edge data present).
    stub = ShellPanelResultants(Nx=0.0, Ny=0.0, Nxy=0.0, Mx=0.0, My=0.0, Mxy=0.0)
    stub_results = [[stub, stub] for _ in panels]
    pair = check_panel_equilibrium(
        stub_results, panels, all_panel_diagnostics=diags, endpoint_tol=1e-6,
    )
    cluster = check_cluster_equilibrium(
        panels, all_panel_diagnostics=diags, endpoint_tol=1e-6,
    )

    def _max_key(rows: list[dict], key: str, boundary: str | None = None) -> float:
        vals = []
        for r in rows:
            if boundary is not None and r.get("boundary_type") != boundary:
                continue
            if key in r and np.isfinite(r[key]):
                vals.append(float(r[key]))
        return float(max(vals)) if vals else float("nan")

    return {
        "mode": interface_constraint_mode,
        "n_elem": float(n_elements_per_panel),
        "ss_dTx": _max_key(pair, "dTx_rel", "skin-skin"),
        "sw_dTx": _max_key(pair, "dTx_rel", "skin-web"),
        "ss_dT_yz": _max_key(pair, "dT_yz_rel", "skin-skin"),
        "sw_dT_yz": _max_key(pair, "dT_yz_rel", "skin-web"),
        "cl_Tx": _max_key(cluster, "Tx_rel_cluster"),
        "cl_Tyz": _max_key(cluster, "T_yz_rel_cluster"),
        "n_pair": float(len(pair)),
        "n_cluster": float(len(cluster)),
    }


def run_mesh_sweep(
    panels: list[Any],
    *,
    n_elem_list: tuple[int, ...] = (8, 12, 16, 24),
    modes: tuple[str, ...] = ("shared", "transformed_basis"),
    out_csv: Path | None = None,
) -> list[dict[str, float]]:
    """Sweep mesh density × interface mode; optionally write ``mesh_sweep_secondary.csv``."""
    rows: list[dict[str, float]] = []
    for mode in modes:
        for ne in n_elem_list:
            rows.append(
                secondary_residuals_for_mode(
                    panels,
                    interface_constraint_mode=mode,
                    n_elements_per_panel=ne,
                )
            )
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "mode", "n_elem", "ss_dTx", "sw_dTx", "ss_dT_yz", "sw_dT_yz",
            "cl_Tx", "cl_Tyz", "n_pair", "n_cluster",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return rows


def shell_mode_sweep_enabled() -> bool:
    return os.environ.get("SHELL_MODE_SWEEP", "").strip() in {"1", "true", "True", "yes"}


def main() -> None:
    from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section

    pb = build_pipeline_b_section("2412", [0.35], chord_m=1.0)
    out = Path(__file__).resolve().parents[2] / "examples" / "output" / "mesh_sweep_secondary.csv"
    rows = run_mesh_sweep(pb.panels, out_csv=out)
    print(f"wrote {out} ({len(rows)} rows)")
    for r in rows:
        print(
            f"{r['mode']:20s} n={int(r['n_elem']):3d}  "
            f"sw_dTx={r['sw_dTx']:.3f}  cl_Tx={r['cl_Tx']:.3f}"
        )


if __name__ == "__main__":
    main()
