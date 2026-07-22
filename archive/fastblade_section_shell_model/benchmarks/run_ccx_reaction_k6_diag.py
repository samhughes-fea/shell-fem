"""Diagnostic-only: ccx reaction K6 vs MITC4 energy K6 (not production).

Usage:
  python -m blade_precompute.section_shell_model.benchmarks.run_ccx_reaction_k6_diag

Requires ``ccx`` on PATH. Leaves the hybrid MITC4-K6 production path unchanged.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section
from blade_precompute.section_shell_model.calculix_ccx.mesh import build_ccx_section_mesh
from blade_precompute.section_shell_model.calculix_ccx.periodic import assemble_k6_from_steps
from blade_precompute.section_shell_model.calculix_ccx.results import parse_ccx_dat_periodic
from blade_precompute.section_shell_model.calculix_ccx.section_export import (
    periodic_ref_node_ids,
    write_section_inp_periodic,
)
from blade_precompute.section_shell_model.engine.homogenisation import (
    compute_elastic_centroid_from_panels,
)
from blade_precompute.section_shell_model.validation.periodic_k6_oracle import (
    mitc4_periodic_k6,
    mitc4_periodic_k6_global,
)

_MODE_NAMES = ("eps0", "ky", "kz", "T", "Vz", "Vy")
_NE = 4


def _rectangle_pb():
    """Trivial two-panel L-section (geometry falsifier; not airfoil-specific)."""
    from types import SimpleNamespace

    from blade_precompute.section_shell_model.pre_processing.constitutive.default_laminates import (
        WallLaminate,
    )

    lam = WallLaminate(E=70e9, t=0.01, nu=0.33, name="iso", n_plies=1)

    class _Panel:
        def __init__(self, label, nodes):
            self.label = label
            self.nodes = np.asarray(nodes, dtype=float)
            ds = np.maximum(
                np.hypot(np.diff(self.nodes[:, 0]), np.diff(self.nodes[:, 1])), 1e-30
            )
            self.s = np.concatenate([[0.0], np.cumsum(ds)])
            self.lam = lam
            self.cell_id = 0
            self.end_boom = None

    panels = [
        _Panel("skin:lower", [[0.0, 0.0], [1.0, 0.0]]),
        _Panel("web:1", [[1.0, 0.0], [1.0, 0.5]]),
    ]
    return SimpleNamespace(panels=panels)


def _run_ccx_steps(pb, n_elements_per_panel: int, work: Path, ccx_exe: str = "ccx"):
    panels = pb.panels
    mesh = build_ccx_section_mesh(
        panels, n_elements_per_panel, interface_constraint_mode="shared"
    )
    y_e, z_e = compute_elastic_centroid_from_panels(panels)
    ref_ids = periodic_ref_node_ids(mesh)
    inp = work / "section_periodic.inp"
    write_section_inp_periodic(mesh, panels, inp, y_e=y_e, z_e=z_e)
    import subprocess

    subprocess.run(
        [shutil.which(ccx_exe), inp.stem],
        cwd=str(work),
        check=True,
        capture_output=True,
        text=True,
    )
    steps = parse_ccx_dat_periodic(work / f"{inp.stem}.dat")
    k6 = assemble_k6_from_steps(steps, ref_ids)
    return k6, steps, ref_ids, float(mesh.l_x)


def _print_diagonal_table(label: str, k6_ccx: np.ndarray, k6_g: np.ndarray, k6_l: np.ndarray):
    print(f"\n=== {label}: diagonals ===")
    print(
        f"{'mode':6} {'ccx':>14} {'mitc4_g':>14} {'ratio':>10} "
        f"{'mitc4_l':>14} {'loc/glb':>10}"
    )
    for i, name in enumerate(_MODE_NAMES):
        cg, mg, ml = float(k6_ccx[i, i]), float(k6_g[i, i]), float(k6_l[i, i])
        ratio = cg / mg if abs(mg) > 0.0 else float("nan")
        loc_glb = ml / mg if abs(mg) > 0.0 else float("nan")
        print(
            f"{name:6} {cg:14.6e} {mg:14.6e} {ratio:10.3f} "
            f"{ml:14.6e} {loc_glb:10.4f}"
        )


def _print_columns(k6_ccx: np.ndarray, k6_g: np.ndarray):
    print("\n=== column-wise: reaction col vs MITC4_g K6[:,m] ===")
    for m, name in enumerate(_MODE_NAMES):
        c = k6_ccx[:, m]
        g = k6_g[:, m]
        relerr = float(np.linalg.norm(c - g) / (np.linalg.norm(g) + 1e-30))
        scale = float((c @ g) / (g @ g + 1e-30))
        off = float(np.max(np.abs(c)) - abs(c[m]))
        print(
            f"mode {m} ({name}): relerr={relerr:.3e}  best_scale={scale:.3f}  "
            f"|offdiag|_max_vs_diag={off:.3e}"
        )
        print(f"  ccx  col: {np.array2string(c, precision=4, suppress_small=True)}")
        print(f"  mitc4col: {np.array2string(g, precision=4, suppress_small=True)}")


def _print_rf_components(steps, ref_ids):
    print("\n=== per-step RF (all components) at ref nodes ===")
    idx = {rid: i for i, rid in enumerate(ref_ids)}
    for step in steps:
        print(f"mode {step.mode} ({_MODE_NAMES[step.mode]}):")
        for rid in ref_ids:
            rf = step.rf_by_node.get(rid, (0.0, 0.0, 0.0))
            print(
                f"  ref[{idx[rid]}] nid={rid}: "
                f"RF=({rf[0]:.6e}, {rf[1]:.6e}, {rf[2]:.6e})"
            )


def diagnose_fixture(label: str, pb, n_elem: int = _NE) -> dict:
    k6_g = mitc4_periodic_k6_global(pb, n_elements_per_panel=n_elem)
    k6_l = mitc4_periodic_k6(pb, n_elements_per_panel=n_elem, dof_frame="local")
    with tempfile.TemporaryDirectory() as tmp:
        k6_c, steps, ref_ids, l_x = _run_ccx_steps(pb, n_elem, Path(tmp))

    _print_diagonal_table(label, k6_c, k6_g, k6_l)
    print(f"\ncell length L_x = {l_x:.6e}")
    _print_columns(k6_c, k6_g)
    _print_rf_components(steps, ref_ids)

    # Sign / DOF check: assemble from rf[0] only vs max |rf| component
    k6_from_fy = np.zeros((6, 6))
    k6_from_fz = np.zeros((6, 6))
    idx = {rid: i for i, rid in enumerate(ref_ids)}
    for step in steps:
        for rid, rf in step.rf_by_node.items():
            if rid in idx:
                k6_from_fy[idx[rid], step.mode] = float(rf[1])
                k6_from_fz[idx[rid], step.mode] = float(rf[2])
    print("\n=== RF DOF check: max |diag| from fx / fy / fz ===")
    for name, mat in (("fx(=prod)", k6_c), ("fy", k6_from_fy), ("fz", k6_from_fz)):
        diags = np.abs(np.diag(0.5 * (mat + mat.T) if name != "fx(=prod)" else mat))
        print(f"  {name}: {np.array2string(diags, precision=4)}")

    ratios = {
        _MODE_NAMES[i]: float(k6_c[i, i] / k6_g[i, i]) if abs(k6_g[i, i]) > 0 else float("nan")
        for i in range(6)
    }
    loc_glb = {
        _MODE_NAMES[i]: float(k6_l[i, i] / k6_g[i, i]) if abs(k6_g[i, i]) > 0 else float("nan")
        for i in range(6)
    }
    return {
        "label": label,
        "L_x": l_x,
        "ratios_ccx_over_mitc4_g": ratios,
        "loc_over_glb": loc_glb,
        "k6_ccx": k6_c,
        "k6_g": k6_g,
    }


def main() -> int:
    if shutil.which("ccx") is None:
        print("CalculiX ccx not found on PATH; aborting diagnostic.", file=sys.stderr)
        return 2

    print("CCX reaction K6 diagnostic (research spike; hybrid production path unchanged)")
    naca = build_pipeline_b_section("2412", [0.35], chord_m=1.0)
    r1 = diagnose_fixture(f"NACA2412+spar NE={_NE}", naca, _NE)
    r2 = diagnose_fixture(f"L-strip NE={_NE}", _rectangle_pb(), _NE)

    print("\n=== SUMMARY diagonal ratios (ccx / mitc4_global) ===")
    for r in (r1, r2):
        print(f"{r['label']}:")
        for name, ratio in r["ratios_ccx_over_mitc4_g"].items():
            print(f"  {name:6} {ratio:10.3f}")
        print(f"  MITC4 local/global diag ratios: {r['loc_over_glb']}")

    print(
        "\nHypothesis check: if bending/torsion ratios >> 1 and RF fy/fz near zero on "
        "ref nodes, RF_x are MPC Lagrange forces (not beam conjugates)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
