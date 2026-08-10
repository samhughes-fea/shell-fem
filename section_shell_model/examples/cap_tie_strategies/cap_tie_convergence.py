"""Does tying the caps fix the BL-003 divergence? (mesh-refinement sweep)

Coverage told us every cap *gets* a load path; it could not tell us whether the
path is any good. This is the test that can: refine the mesh and watch cap
in-plane shear. A structurally connected component converges; a floating one
does not, because its shear is set by its own strip stiffness rather than by
equilibrium with the rest of the section.

Runs the same section at increasing density with cap ties off and on, and
reports peak ``|Nxy|`` per component kind. Skin and web are the control: they
are connected under both settings and must converge either way.

Run from the repository root::

    python blade_structure_utilities/fem_utilities/section_shell_model/examples/\\
        cap_tie_strategies/cap_tie_convergence.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (  # noqa: E402
    BeamResultantVector,
)
from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (  # noqa: E402
    build_section_view,
    resolve_system_type,
)
from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import (  # noqa: E402
    AirfoilSDF,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.api.shell_panel_topology import (  # noqa: E402
    shell_panel_topology_from_shell_inputs,
)
from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.section_linear_static_runner import (  # noqa: E402
    SectionLinearStaticRunner,
)
from blade_structure_utilities.mesh_utilities.section_mesh._section_to_shell_inputs import (
    shell_mesh_inputs_from_section,
)
from blade_structure_utilities.mesh_utilities.section_mesh import (  # noqa: E402
    build_section_strip_mesh,
)

LAYOUT = "5D-F"
CHORD = 0.241
TWIST_DEG = 34.1
# Element length as a fraction of chord; each step roughly halves it.
REFINEMENTS = [0.024, 0.016, 0.012, 0.008, 0.006]
THICKNESS = {"skin": 0.001, "spar": 0.004, "web": 0.0015}
# A torsion-dominated load so in-plane shear is the governing response.
LOAD = {"T": 5.0e3, "Vy": 2.0e4, "Mz": 1.0e4}


def _kind_of(label: str) -> str:
    return label.split(":", 1)[0] if ":" in label else "panel"


def _build(frac: float):
    twist = float(np.deg2rad(TWIST_DEG))
    airfoil = AirfoilSDF.from_naca_series(
        6, 63.0, 4.0, 15.0, n_points=200, chord=CHORD, closed_te=True
    )
    section = build_section_view(
        airfoil,
        resolve_system_type(LAYOUT),
        twist_angle_rad=twist,
        skin_thickness=THICKNESS["skin"],
        web_thickness=THICKNESS["web"],
        cap_height=THICKNESS["spar"],
    )
    shell = shell_mesh_inputs_from_section(
        section, twist_rad=twist, layout_key=LAYOUT, n_web_samples=20, n_cap_samples=80
    )
    mesh = build_section_strip_mesh(shell, target_element_length_frac=frac, endpoint_tol=1e-6)
    return shell_panel_topology_from_shell_inputs(shell), mesh


def _peak_nxy_by_kind(runner) -> dict[str, float]:
    """Peak |Nxy| over the nodes of each component kind."""
    sec = runner.secondary_results
    nodal = getattr(sec, "nodal_results", None) if sec is not None else None
    fields = getattr(nodal, "fields", None) or {}
    nxy = np.asarray(fields.get("Nxy", []), dtype=float)
    if nxy.size == 0:
        raise RuntimeError(
            "nodal Nxy is empty; available secondary nodal fields: "
            f"{sorted(fields.keys())}"
        )
    out: dict[str, float] = {}
    for pm in runner.assembly.panel_maps:
        if not len(pm.s_nodes):
            continue
        kind = _kind_of(pm.panel_label)
        gns = [g for g in pm.global_nodes if 0 <= g < nxy.size]
        if not gns:
            continue
        peak = float(np.nanmax(np.abs(nxy[np.asarray(gns, dtype=int)])))
        out[kind] = max(out.get(kind, 0.0), peak)
    return out


def _run(frac: float, cap_tie_mode: str, channel: str):
    panel_topology, mesh = _build(frac)
    resultants = BeamResultantVector.from_named(**LOAD)
    kwargs: dict = {}
    if channel == "thinwall":
        # Only the thinwall channel actually loads the caps: equipollent drops
        # every strip whose kind is not skin/web, so under it a cap is moved
        # purely by compatibility with the skin.
        from blade_structure_utilities.load_utilities.beam_to_shell_loads.shape_external import (
            shape_section_force_vector,
        )

        kwargs["section_force_vector"] = shape_section_force_vector(
            resultants, panel_topology.shell_inputs.midlines
        )
    runner = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        section_strip_mesh=mesh,
        resultants=resultants,
        cap_tie_mode=cap_tie_mode,
        load_channel=channel,
        compute_gaussian=True,
        compute_nodal=True,
        **kwargs,
    )
    runner.execute()
    return runner, int(runner.assembly.node_meta.n_nodes)


def main() -> int:
    print(f"BL-003 cap-tie convergence — layout {LAYOUT}, chord {CHORD} m, "
          f"twist {TWIST_DEG} deg")
    print(f"load {LOAD}\n")

    rows: dict[tuple[str, str], list] = {}
    for channel in ("thinwall", "equipollent"):
        for mode in ("none", "skin_offset"):
            print(f"--- channel = {channel},  cap_tie_mode = {mode} ---")
            rows[(channel, mode)] = []
            for frac in REFINEMENTS:
                try:
                    runner, n_nodes = _run(frac, mode, channel)
                except Exception as exc:
                    print(f"  frac={frac:.4f}  FAILED: "
                          f"{type(exc).__name__}: {str(exc)[:120]}")
                    continue
                peaks = _peak_nxy_by_kind(runner)
                rows[(channel, mode)].append((frac, n_nodes, peaks))
                rep = getattr(runner, "cap_tie_report", None)
                extra = (
                    f"  [{rep.n_links} links]"
                    if rep is not None and rep.mode != "none"
                    else ""
                )
                print(
                    f"  frac={frac:.4f}  nodes={n_nodes:5d}  "
                    f"skin={peaks.get('skin', float('nan')):11.4e}  "
                    f"web={peaks.get('web', float('nan')):11.4e}  "
                    f"cap={peaks.get('cap', float('nan')):11.4e}{extra}"
                )
            print()

    print("=== drift over the refinement sweep (max/min of peak |Nxy|) ===")
    hdr = f"{'channel':12s} {'mode':13s} {'kind':6s} {'coarsest':>12s} {'finest':>12s} {'max/min':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for (channel, mode), data in rows.items():
        if len(data) < 2:
            continue
        for kind in ("skin", "web", "cap"):
            vals = [p.get(kind) for _, _, p in data if p.get(kind) is not None]
            if len(vals) < 2:
                continue
            lo, hi = min(vals), max(vals)
            ratio = hi / lo if lo > 0 else float("inf")
            print(f"{channel:12s} {mode:13s} {kind:6s} {vals[0]:12.4e} "
                  f"{vals[-1]:12.4e} {ratio:9.2f}x")

    print(
        "\nReading: skin and web are connected either way and set the scale for "
        "'converged'.\nIf cap drift falls to their level once ties are on, the "
        "cap is carrying load through\nthe section rather than floating. "
        "'thinwall' is the production channel and the\nonly one that applies "
        "load to the caps at all."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
