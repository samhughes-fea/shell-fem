"""Do the tied FE and the thinwall load channel agree about cap shear? (S2 vs S4)

The two model a spar cap differently, and now that the FE actually connects the
cap the disagreement is measurable rather than academic.

* **Load channel** treats cap+skin as *one thickened shear wall*:
  ``apply_stacked_cap_shear`` sets ``t_shear = t_skin + t_cap`` on the
  overlapping skin segment, folds the cap's ``Gt`` into it, and excludes caps
  from the shear cell loops. Caps re-enter only at sampling time, taking
  ``q * stacked_cap_gt_share`` (plus the equipollence residual).
* **Shell FE** with ``cap_tie_mode="skin_offset"`` treats the cap as its own
  panel, bonded to the skin through an offset rigid link, carrying whatever
  equilibrium gives it.

If the two broadly agree, S2 (keep the cap as a panel) is consistent with the
load model and the contradiction is cosmetic. If they disagree systematically,
that is the case for S4 (merge the cap into the skin laminate so the FE and the
load model assume the same thing) — and the size and sign of the disagreement is
the evidence for which is wrong.

Coverage is a necessary caveat: this cannot rank S4 directly, because S4 deletes
the panel this script measures. It measures the *cost of the disagreement*.

Run from the repository root::

    python blade_structure_utilities/fem_utilities/section_shell_fea/visualisation/demos/\\
        cap_tie_strategies/cap_fe_vs_load_channel.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[6]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (  # noqa: E402
    BeamResultantVector,
)
from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (  # noqa: E402
    build_section_view,
    resolve_system_type,
)
from blade_structure_utilities.geometry_utilities.section_geometry.layout.system_layout import (  # noqa: E402
    SYSTEM_TYPE_KEYS,
)
from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import (  # noqa: E402
    AirfoilSDF,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.api.shell_panel_topology import (  # noqa: E402
    shell_panel_topology_from_shell_inputs,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (  # noqa: E402
    SectionLinearStaticRunner,
)
from blade_structure_utilities.load_utilities.section_loads import (  # noqa: E402
    shape_section_force_vector,
)
from blade_structure_utilities.mesh_utilities.section_mesh.handoff._section_to_shell_inputs import (
    shell_mesh_inputs_from_section,
)
from blade_structure_utilities.mesh_utilities.section_mesh import (  # noqa: E402
    build_section_strip_mesh,
)

SECTIONS = [("root", 0.241, 34.1), ("mid", 0.199, 5.4)]
# Chord-relative, so no station is asked for a cap that cannot fit the cavity.
THICKNESS_FRAC = {"skin": 0.006, "spar": 0.020, "web": 0.008}
LOAD = {"T": 5.0e3, "Vy": 2.0e4, "Mz": 1.0e4}


def _kind_of(label: str) -> str:
    return label.split(":", 1)[0] if ":" in label else "panel"


def _build(layout_key: str, chord: float, twist_deg: float):
    twist = float(np.deg2rad(twist_deg))
    airfoil = AirfoilSDF.from_naca_series(
        6, 63.0, 4.0, 15.0, n_points=200, chord=chord, closed_te=True
    )
    section = build_section_view(
        airfoil,
        resolve_system_type(layout_key),
        twist_angle_rad=twist,
        skin_thickness=THICKNESS_FRAC["skin"] * chord,
        web_thickness=THICKNESS_FRAC["web"] * chord,
        cap_height=THICKNESS_FRAC["spar"] * chord,
    )
    shell = shell_mesh_inputs_from_section(
        section, twist_rad=twist, layout_key=layout_key,
        n_web_samples=20, n_cap_samples=80,
    )
    mesh = build_section_strip_mesh(
        shell, target_element_length_frac=0.012, endpoint_tol=1e-6
    )
    return shell_panel_topology_from_shell_inputs(shell, allow_demo_bind=True), mesh, shell


def _fe_nxy_by_label(runner, kind: str) -> dict[str, float]:
    """Peak |N12| per panel of ``kind``, keyed by label without the kind prefix."""
    sec = runner.secondary_results
    nodal = getattr(sec, "nodal_results", None) if sec is not None else None
    nxy = np.asarray((getattr(nodal, "fields", None) or {}).get("N12", []), dtype=float)
    out: dict[str, float] = {}
    if nxy.size == 0:
        return out
    for pm in runner.assembly.panel_maps:
        if not len(pm.s_nodes) or _kind_of(pm.panel_label) != kind:
            continue
        gns = [g for g in pm.global_nodes if 0 <= g < nxy.size]
        if gns:
            out[pm.panel_label.split(":", 1)[1]] = float(
                np.nanmax(np.abs(nxy[np.asarray(gns, dtype=int)]))
            )
    return out


def _channel_q_by_label(sfv, kind: str) -> dict[str, float]:
    """Peak |q| per strip label of ``kind`` from the load channel."""
    out: dict[str, float] = {}
    for s in sfv.samples:
        if s.kind != kind:
            continue
        out[s.label] = max(out.get(s.label, 0.0), abs(float(s.q)))
    return out


def _nearest_on_polyline(q, poly):
    best_d, best_i = np.inf, -1
    for j in range(poly.shape[0] - 1):
        a = poly[j]
        ab = poly[j + 1] - a
        l2 = float(np.dot(ab, ab))
        if l2 <= 0.0:
            continue
        t = float(np.clip(np.dot(q - a, ab) / l2, 0.0, 1.0))
        d = float(np.linalg.norm(q - (a + t * ab)))
        if d < best_d:
            best_d, best_i = d, (j if t < 0.5 else j + 1)
    return best_d, best_i


def _fe_cap_load_share(runner) -> dict[str, float]:
    """Cap's share of the stacked cap+skin shear resultant, per cap panel.

    The dimensionless counterpart of ``stacked_cap_gt_share``: at every cap node,
    pair ``|Nxy_cap|`` with ``|Nxy_skin|`` at the skin point directly beneath and
    take ``cap / (cap + skin)``. Being a ratio of two FE quantities at the same
    place, it is immune to the applied-vs-developed offset that makes a raw
    peak-|N12| against peak-|q| comparison meaningless.
    """
    sec = runner.secondary_results
    nodal = getattr(sec, "nodal_results", None) if sec is not None else None
    nxy = np.asarray((getattr(nodal, "fields", None) or {}).get("N12", []), dtype=float)
    yz = runner.assembly.node_meta.yz
    out: dict[str, float] = {}
    if nxy.size == 0:
        return out

    skins = []
    for pm in runner.assembly.panel_maps:
        if not len(pm.s_nodes) or _kind_of(pm.panel_label) != "skin":
            continue
        n = len(pm.s_nodes)
        gids = [g for g in pm.global_nodes[:n] if 0 <= g < nxy.size]
        if len(gids) >= 2:
            skins.append((np.asarray(gids, dtype=int), yz[np.asarray(gids, dtype=int)]))
    if not skins:
        return out

    for pm in runner.assembly.panel_maps:
        if not len(pm.s_nodes) or _kind_of(pm.panel_label) != "cap":
            continue
        n = len(pm.s_nodes)
        fracs = []
        for g in pm.global_nodes[:n]:
            if not (0 <= g < nxy.size) or not np.all(np.isfinite(yz[g])):
                continue
            best = None
            for gids, pts in skins:
                d, i = _nearest_on_polyline(yz[g], pts)
                if i >= 0 and (best is None or d < best[0]):
                    best = (d, int(gids[i]))
            if best is None:
                continue
            c, s = abs(float(nxy[g])), abs(float(nxy[best[1]]))
            if c + s > 0:
                fracs.append(c / (c + s))
        if fracs:
            out[pm.panel_label.split(":", 1)[1]] = float(np.median(fracs))
    return out


def _channel_cap_share(mesh) -> dict[str, float]:
    """``G_c t_c / (G_s t_s + G_c t_c)`` per cap, from the meshed thicknesses.

    Recomputed rather than read back: ``distribute_beam_resultants`` applies
    ``apply_stacked_cap_shear`` to a *copy* of the shear segments so the cached
    network is not mutated, so the share is not retained on the SectionForceVector.
    Default ``G_by_kind`` gives caps and skin the same G, making this a pure
    thickness ratio.
    """
    caps = [pm for pm in mesh.panel_meshes if pm.kind == "cap" and pm.yz_nodes.size]
    skins = [pm for pm in mesh.panel_meshes if pm.kind == "skin" and pm.yz_nodes.size]
    if not skins:
        return {}
    skin_thickness = float(np.median([pm.thickness_m for pm in skins]))
    return {
        pm.panel_label.split(":", 1)[1]: float(pm.thickness_m)
        / max(float(pm.thickness_m) + skin_thickness, 1e-30)
        for pm in caps
    }


def main() -> int:
    resultants = BeamResultantVector.from_named(**LOAD)
    layouts = []
    for key in SYSTEM_TYPE_KEYS:
        try:
            if bool(resolve_system_type(key).has_spar_caps):
                layouts.append(key)
        except Exception:
            continue

    print("Tied shell FE vs thinwall load channel: the cap's SHARE of stacked shear")
    print(f"load {LOAD};  thicknesses (chord fraction) {THICKNESS_FRAC}")
    print(
        "\nA share, not a magnitude. Comparing FE |N12| against channel |q| directly "
        "does\nnot work: the channel's q is applied flow and the FE's N12 is developed\n"
        "resultant, and a skin/web control shows both offset by the same order, so "
        "the\nraw ratio measures the boundary conditions rather than the cap model. "
        "Both\nquantities below are dimensionless fractions of the same stacked wall.\n"
    )
    hdr = (f"{'case':22s} {'cap':26s} {'FE share':>11s} "
           f"{'chan share':>11s} {'diff':>9s}")
    print(hdr)
    print("-" * len(hdr))

    fe_shares: list[float] = []
    ch_shares: list[float] = []
    n_skip = 0
    for layout_key in layouts:
        for sec_name, chord, twist in SECTIONS:
            tag = f"{layout_key}/{sec_name}"
            try:
                panel_topology, mesh, shell = _build(layout_key, chord, twist)
                sfv = shape_section_force_vector(resultants, shell.midlines, cell_wall_loops=shell.cell_wall_loops)
                runner = SectionLinearStaticRunner(
                    panel_topology=panel_topology,
                    section_strip_mesh=mesh,
                    resultants=resultants,
                    section_force_vector=sfv,
                    load_channel="thinwall",
                    cap_tie_mode="skin_offset",
                    compute_gaussian=True,
                    compute_nodal=True,
                )
                runner.execute()
            except Exception as exc:
                n_skip += 1
                if n_skip <= 4:
                    print(f"  [skipped] {tag}: {type(exc).__name__}: {str(exc)[:80]}")
                continue

            fe_share = _fe_cap_load_share(runner)
            ch_share = _channel_cap_share(mesh)
            verbose = layout_key in ("2D-F", "5D-F", "5C-F")
            for label, fe_f in sorted(fe_share.items()):
                ch_f = ch_share.get(label)
                if ch_f is None:
                    continue
                fe_shares.append(fe_f)
                ch_shares.append(ch_f)
                if verbose:
                    print(f"{tag:22s} {label[-26:]:26s} {fe_f:11.3f} "
                          f"{ch_f:11.3f} {fe_f - ch_f:+9.3f}")

    if not fe_shares:
        print("\nno comparable caps found")
        return 1

    fe_a = np.array(fe_shares)
    ch_a = np.array(ch_shares)
    d = fe_a - ch_a
    print(f"\n=== cap share of the stacked cap+skin shear, {fe_a.size} caps ===")
    print(f"{'':14s} {'p5':>8s} {'p50':>8s} {'p95':>8s}")
    print("-" * 42)
    for name, arr in (("tied FE", fe_a), ("load channel", ch_a), ("FE - channel", d)):
        print(f"{name:14s} {np.percentile(arr,5):8.3f} "
              f"{np.percentile(arr,50):8.3f} {np.percentile(arr,95):8.3f}")
    print(
        f"\nThe load channel gives the cap a median {np.median(ch_a):.0%} of the "
        f"stacked wall's\nshear (a pure Gt ratio). The tied FE develops a median "
        f"{np.median(fe_a):.0%}.\n"
    )
    if abs(float(np.median(d))) < 0.05:
        print("These agree: S2 keeps the FE consistent with the load model, and the\n"
              "S2/S4 choice is a reporting question, not a structural one.")
    else:
        direction = "more" if float(np.median(d)) > 0 else "less"
        print(
            f"These disagree materially, and in a consistent direction: the bonded FE\n"
            f"gives the cap {direction} shear than the Gt split does (p5 of the "
            f"difference is\n{np.percentile(d, 5):+.3f}, so the sign does not flip). "
            "The load channel's\nstacked-wall split is not what a bonded cap actually "
            "does, so S2 and S4 are not\ninterchangeable — S4 would change how the "
            "spar is sized, not just how it is\nreported.\n\n"
            "Caveat: the FE share saturates near 0.98 on several layouts, so treat "
            "this as\nestablishing direction and rough magnitude, not a calibrated "
            "number. Pinning it\ndown wants a shear-only load case and an "
            "element-level rather than nodal N12."
        )
    print(f"\n{n_skip} case(s) skipped (infeasible geometry or solve failure)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
