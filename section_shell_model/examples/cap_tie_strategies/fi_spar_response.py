"""Why does fi_spar wander when you thicken the spar cap? (BL-003 part 3)

Sweeping cap thickness gives a non-monotone ``fi_spar``. Since it now drives
governing-DV selection in the sizing loop, a driver that wanders when you
thicken the thing it measures is not safe to size against.

``fi_by_component["spar"]`` is four nested maxima — over the 4 Hashin modes, over
plies, over elements, over the cap panels — so one element, or one ply of one
element, sets the number. This script asks *which* one, at every thickness. If
the argmax location hops between steps, the wander is an argmax artefact rather
than a change in the underlying field.

Three suspects, separated by the ``--mode`` runs:

* **mesh**: changing ``cap_height`` moves the cap midline (normal depth
  ``t_skin + cap_height/2``), which changes the cap's arc length, its element
  count (``int(round(arc/target))`` is a step function), and — because the skin
  ring is split at cap endpoints — the skin's nodes too.
* **the ramp element**: a C-family cap endpoint not on a web is snapped onto the
  *skin* midline while the interior sits at cap depth, so the end element is a
  distorted normal-direction ramp whose length grows with ``cap_height/2``. The
  mesh policy's own docstring records that sliver cap elements "report in-plane
  shear orders of magnitude above their neighbours -- enough to dominate a
  section's Hashin envelope".
* **physics**: every cap ply is 0 deg with identical strengths, so FI varies
  across plies only through ``z`` in ``eps = eps0 + z*kappa``. Max-over-plies
  always lands on the outer ply at ``z = +/-t/2``, whose ``|z|`` grows with
  thickness while membrane ``N/t`` falls. That term is real and must not be
  designed away.

Run from the repository root::

    python blade_structure_utilities/fem_utilities/section_shell_model/examples/\\
        cap_tie_strategies/fi_spar_response.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from blade_structure_precompute.module.section_selection.processing.shell_station.evaluate import (  # noqa: E402
    evaluate_station,
)

LAYOUT = "2C-CN"
T_SKIN = 0.006
T_WEB = 0.004
T_SPARS = [0.004, 0.008, 0.012, 0.020, 0.030]
LOADS = {"M_flapwise": 5.0e5, "V_flapwise": 1.0e5}
# Not 10: effective_n_segments_spec treats the default_n sentinel as "unset" and
# silently falls back to the length policy, so a fixed count of 10 is not fixed.
FIXED_N = 12


class _Iso:
    E = 70e9
    G = 27e9
    density = 2700.0

    def get_E(self) -> float:
        return self.E


def _airfoil(chord: float = 1.0, t_c: float = 0.12, n: int = 120) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack(
        [0.5 * chord * (1.0 + np.cos(th)), 0.5 * chord * t_c * np.sin(th)]
    )


def _run(t_spar: float, *, cap_tie_mode: str, n_elements: int | None):
    mats = {"skin": _Iso(), "spar": _Iso(), "web": _Iso()}
    return evaluate_station(
        coords=_airfoil(),
        metadata={"chord": 1.0, "twist_deg": 0.0},
        bsp_catalog_key=LAYOUT,
        t_skin=T_SKIN,
        t_spar=t_spar,
        t_web=T_WEB,
        component_materials=mats,
        run_shell_fi=True,
        extract_full_shell_payload=True,
        cap_tie_mode=cap_tie_mode,
        n_elements_per_panel=n_elements,
        **LOADS,
    )


def _cap_argmax(result):
    """Locate the element+ply that sets cap FI, and describe the mesh around it."""
    rows = result.full_element_results or []
    caps = [r for r in rows if str(r.get("panel_label", "")).startswith("cap")]
    if not caps:
        return None

    # Elements per cap panel, so "is this an end element?" is answerable.
    n_by_panel: dict[str, int] = {}
    for r in caps:
        lab = str(r["panel_label"])
        n_by_panel[lab] = max(n_by_panel.get(lab, 0), int(r["local_element_index"]) + 1)

    best = None
    for r in caps:
        q = r.get("quaternary") or {}
        fi_plies = q.get("hashin_in_plane")
        if not fi_plies:
            continue
        arr = np.asarray(fi_plies, dtype=float)
        if arr.size == 0 or not np.any(np.isfinite(arr)):
            continue
        k = int(np.nanargmax(arr))
        v = float(arr[k])
        if best is None or v > best["fi"]:
            lab = str(r["panel_label"])
            lei = int(r["local_element_index"])
            n_el = n_by_panel[lab]
            best = {
                "fi": v,
                "panel": lab,
                "elem": lei,
                "n_elem": n_el,
                "ply": k,
                "n_ply": int(arr.size),
                "is_end": lei in (0, n_el - 1),
            }
    if best is not None:
        best["n_cap_elems"] = sum(n_by_panel.values())
        best["n_rows"] = len(rows)
    return best


def _frozen_mesh_sweep(cap_tie_mode: str = "skin_offset") -> list[dict]:
    """Vary ONLY the laminate: one mesh, built once, reused at every thickness.

    ``evaluate_station`` cannot do this — its prep-cache key includes ``t_spar``,
    so every step is a genuine geometry+mesh rebuild. Driving the runner directly
    with a prebuilt ``SectionStripMesh`` is the only way to hold the mesh still.
    Whatever wander survives here would be physics, not meshing.

    **This run is currently INCONCLUSIVE and self-reports as such.** Panel
    thickness is read from the midline strip (``shell_panel_bind.py:126``,
    ``float(m.thickness_m)``), not from the ``t_spar`` kwarg — that kwarg only
    selects the laminate. Freezing ``shell_inputs`` to freeze the mesh therefore
    also freezes the cap's *geometric* thickness, so the sweep varies nothing
    that reaches the FE and every step returns an identical answer. A valid
    version has to vary ``midlines[cap].thickness_m`` and the matching
    ``StripPanelMesh.thickness_m`` while holding every polyline fixed. The guard
    below fails the run rather than let a flat line be read as "physics is
    monotone".
    """
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.api.shell_panel_topology import (
        ShellPanelTopology,
    )
    from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.section_linear_static_runner import (
        SectionLinearStaticRunner,
    )
    from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
        BeamResultantVector,
    )
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (
        build_section_view,
        resolve_system_type,
    )
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import (
        AirfoilSDF,
    )
    from blade_structure_utilities.load_utilities.beam_to_shell_loads import (
        shape_section_force_vector,
    )
    from blade_structure_utilities.material_utilities.adapters import (
        shell_panels_from_mesh_inputs,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh import (
        build_section_strip_mesh,
        build_shell_mesh_inputs,
    )

    print(f"\n=== FROZEN mesh (built once at t_spar={T_SPARS[0]*1e3:.0f} mm, "
          f"cap_tie_mode={cap_tie_mode}) ===")

    # Geometry + mesh built ONCE, at the reference thickness, then never rebuilt.
    af = AirfoilSDF.from_naca("0012", n_points=200, chord=1.0)
    section = build_section_view(
        af, resolve_system_type(LAYOUT), twist_angle_rad=0.0,
        skin_thickness=T_SKIN, web_thickness=T_WEB, cap_height=T_SPARS[0],
    )
    shell_inputs = build_shell_mesh_inputs(section, twist_rad=0.0, layout_key=LAYOUT)
    mesh = build_section_strip_mesh(shell_inputs, target_element_length_frac=0.012)
    n_nodes_ref = sum(int(pm.yz_nodes.shape[0]) for pm in mesh.panel_meshes)

    mats = {"skin": _Iso(), "spar": _Iso(), "web": _Iso()}
    resultants = BeamResultantVector.from_named(My=LOADS["M_flapwise"],
                                                Vz=LOADS["V_flapwise"])
    hdr = (f"{'t_spar req':>10s} {'fi_spar':>9s} {'max_fi':>9s} "
           f"{'argmax panel':>22s} {'elem':>7s} {'ply':>9s} {'nodes':>6s}")
    print(hdr)
    print("-" * len(hdr))

    out = []
    for t_spar in T_SPARS:
        # Only the laminates change; shell_inputs and mesh are the same objects.
        panels, booms, webs_geom, n_cells = shell_panels_from_mesh_inputs(
            shell_inputs, component_materials=mats,
            t_skin=T_SKIN, t_spar=t_spar, t_web=T_WEB,
        )
        topo = ShellPanelTopology(
            shell_inputs=shell_inputs, panels=list(panels), booms=list(booms),
            webs_geom=list(webs_geom), n_cells=int(n_cells), section=section,
        )
        runner = SectionLinearStaticRunner(
            panel_topology=topo,
            section_strip_mesh=mesh,
            resultants=resultants,
            section_force_vector=shape_section_force_vector(
                resultants, shell_inputs.midlines
            ),
            load_channel="thinwall",
            cap_tie_mode=cap_tie_mode,
        ).execute()
        quat = runner.get_results().quaternary_results
        gov = (quat.governing or {}) if quat is not None else {}
        labels = list(getattr(runner.assembly, "lab_p", None) or [])
        hi = None if quat is None or quat.hashin_fi is None else quat.hashin_fi.in_plane
        # Cap-only max, mirroring _fi_by_component_from_panels.
        fi_spar = 0.0
        if hi is not None:
            pidx = np.asarray(runner.get_results().tertiary_results.panel_index)
            for pi, lab in enumerate(labels):
                if not str(lab).startswith("cap"):
                    continue
                sel = np.where(pidx == pi)[0]
                if sel.size:
                    fi_spar = max(fi_spar, float(np.nanmax(hi[sel])))
        gp = int(gov.get("panel_index", -1))
        gl = labels[gp] if 0 <= gp < len(labels) else "?"
        n_nodes = sum(int(pm.yz_nodes.shape[0]) for pm in mesh.panel_meshes)
        assert n_nodes == n_nodes_ref, "mesh changed — the freeze failed"
        out.append({"t_req": t_spar, "t_real": t_spar, "fi_spar": fi_spar,
                    "max_fi": gov.get("fi"),
                    "argmax": {"panel": str(gl), "elem": int(gov.get("element_index", -1)),
                               "n_elem": 0, "ply": int(gov.get("ply_index", -1)),
                               "n_ply": 0, "is_end": False,
                               "n_cap_elems": 0, "n_rows": 0}})
        print(f"{t_spar*1e3:10.2f} {fi_spar:9.5f} "
              f"{float(gov.get('fi', float('nan'))):9.4f} {str(gl)[-22:]:>22s} "
              f"{int(gov.get('element_index', -1)):7d} "
              f"{int(gov.get('ply_index', -1)):9d} {n_nodes:6d}")

    # A frozen mesh should still show the laminate change. If nothing moved at
    # all, the sweep varied nothing that reaches the FE and the run says nothing
    # about physics -- refuse to report it as a result.
    spread = {round(float(r["fi_spar"]), 12) for r in out}
    if len(spread) <= 1:
        raise RuntimeError(
            "frozen-mesh run is INCONCLUSIVE: fi_spar is identical at every "
            "thickness, so t_spar never reached the FE. Panel thickness comes "
            "from midlines[i].thickness_m, which was frozen along with the mesh. "
            "Vary the strip thickness (geometry AND StripPanelMesh) while holding "
            "the polylines fixed before trusting this run."
        )
    return out


def _sweep(label: str, *, cap_tie_mode: str, n_elements: int | None) -> list[dict]:
    print(f"\n=== {label} (cap_tie_mode={cap_tie_mode}, "
          f"n_elements_per_panel={n_elements}) ===")
    hdr = (f"{'t_spar req':>10s} {'realized':>9s} {'fi_spar':>9s} {'max_fi':>9s} "
           f"{'cap argmax panel':>22s} {'elem':>9s} {'ply':>9s} {'end?':>5s} "
           f"{'capEl':>6s} {'rows':>5s}")
    print(hdr)
    print("-" * len(hdr))
    out = []
    for t_spar in T_SPARS:
        try:
            res = _run(t_spar, cap_tie_mode=cap_tie_mode, n_elements=n_elements)
        except Exception as exc:
            print(f"{t_spar*1e3:10.2f}  FAILED: {type(exc).__name__}: {str(exc)[:70]}")
            continue
        am = _cap_argmax(res)
        fi_spar = (res.fi_by_component or {}).get("spar")
        rec = {"t_req": t_spar, "t_real": res.t_spar, "fi_spar": fi_spar,
               "max_fi": res.max_fi_hashin, "argmax": am}
        out.append(rec)
        if am is None:
            print(f"{t_spar*1e3:10.2f} {res.t_spar*1e3:9.3f} "
                  f"{(fi_spar or float('nan')):9.5f} "
                  f"{(res.max_fi_hashin or float('nan')):9.4f}   (no cap elements)")
            continue
        print(f"{t_spar*1e3:10.2f} {res.t_spar*1e3:9.3f} "
              f"{(fi_spar or float('nan')):9.5f} "
              f"{(res.max_fi_hashin or float('nan')):9.4f} "
              f"{am['panel'][-22:]:>22s} "
              f"{am['elem']:4d}/{am['n_elem']:<4d} "
              f"{am['ply']:4d}/{am['n_ply']:<4d} "
              f"{('YES' if am['is_end'] else 'no'):>5s} "
              f"{am['n_cap_elems']:6d} {am['n_rows']:5d}")
    return out


def _verdict(label: str, recs: list[dict]) -> None:
    good = [r for r in recs if r["argmax"] is not None and r["fi_spar"] is not None]
    if len(good) < 2:
        print(f"  {label}: too few points")
        return
    fi = [r["fi_spar"] for r in good]
    mono = all(fi[i] <= fi[i + 1] + 1e-12 for i in range(len(fi) - 1)) or all(
        fi[i] >= fi[i + 1] - 1e-12 for i in range(len(fi) - 1)
    )
    locs = {(r["argmax"]["panel"], r["argmax"]["elem"]) for r in good}
    plies = {r["argmax"]["ply"] for r in good}
    ends = sum(1 for r in good if r["argmax"]["is_end"])
    counts = {r["argmax"]["n_cap_elems"] for r in good}
    spread = (max(fi) - min(fi)) / max(np.median(fi), 1e-30)
    print(f"  {label}:")
    print(f"     monotone in t_spar        : {mono}")
    print(f"     fi_spar spread (max-min)/med: {spread:.1%}")
    print(f"     distinct argmax elements  : {len(locs)} of {len(good)} steps  {sorted(locs)}")
    print(f"     argmax on a cap END elem  : {ends}/{len(good)} steps")
    print(f"     distinct argmax ply idx   : {sorted(plies)}")
    print(f"     cap element counts seen   : {sorted(counts)}")


def main() -> int:
    print("fi_spar response to cap thickness — "
          f"layout {LAYOUT}, t_skin={T_SKIN*1e3:.1f} mm, loads {LOADS}")

    runs = []
    for label, mode, n_el in (
        ("free mesh, ties ON", "skin_offset", None),
        ("free mesh, ties OFF", "none", None),
        (f"fixed counts (n={FIXED_N}), ties ON", "skin_offset", FIXED_N),
    ):
        runs.append((label, _sweep(label, cap_tie_mode=mode, n_elements=n_el)))

    try:
        runs.append(("FROZEN mesh, ties ON", _frozen_mesh_sweep()))
    except Exception as exc:
        print(f"\n  frozen-mesh run failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    print("\n=== verdicts ===")
    for label, recs in runs:
        _verdict(label, recs)

    print(
        "\nReading: if 'distinct argmax elements' > 1 the wander is an argmax that "
        "hops,\nnot a field that changes. If those hops land on cap END elements, "
        "the cause is the\nramp element created by snapping a cap endpoint onto the "
        "skin midline. If the\nfixed-count run is materially steadier, element-count "
        "steps are contributing.\nWhatever survives a frozen mesh is the physical "
        "outer-ply term and is real."
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
