"""
Smoke repro: ``build_shell_midline_strips`` + ``ShellMidlineStrip``
→ ``build_shell_mesh_inputs`` → ``build_section_strip_mesh``.

This example is **element-agnostic**: it only generates the section geometry and
meshes the centerlines (``build_section_strip_mesh`` is an alias of the theory-agnostic
``build_section_strip_mesh``). No shell element is solved here.

Verifies that ``build_section_strip_mesh`` produces a wire-compatible section centerline
(strip) mesh from the per-subcomponent midlines (skin, webs, caps) emitted by
``build_shell_mesh_inputs``, and that the resulting endpoint clustering
correctly identifies the section's junctions (web/skin and cap/skin
intersections).

Also exercises the explicit geometry-export path via
``build_shell_midline_strips`` (from
``blade_structure_utilities.geometry_utilities.section_geometry.interface.shell_midline_export``) and
asserts that non-skin strips match ``ShellMeshInputs`` (skin may be split into
``seg*`` sub-strips at web/cap junctions; web/cap polylines may be snapped).

Reuses the same run007 / station i000 / NACA63-415 / 2D-F inputs as
``repro_geometry_to_midlines.py`` so the mesh can be visually compared
against the section_geometry baseline.

Outputs (PNG):
    blade_precompute/section_shell_model/examples/output/section_centerline_mesh_v2/
    section_mesh_<layout>_i<idx>.png

Each figure is a **2×2** equal-aspect B-frame taxonomy layout
(Subcomponent → Strip → Panel → Element):

1. **Subcomponents** — ``section_geometry`` SDF, coloured by kind (skin / web / cap).
2. **Strips** — ``ShellMidlineStrip`` midlines + geometry sample vertices from
   ``shell_inputs.midlines``; ``tab20`` per strip (no FE mesh).
3. **Panels** — constitutive patches from ``SectionStripMesh.panel_meshes``:
   continuous discretised contour + nodes; ``tab20`` per panel (same label→colour
   map as strips); no junction rings / Class C rays.
4. **Elements** — full strip mesh (element edges + nodes + junctions) from
   ``build_section_strip_mesh(..., target_element_length_frac=TARGET_ELEMENT_LENGTH_FRAC)``
   (production default ≈ 1.2 % of section ``chord_m``). Each FE contour **element**
   gets its own colour (no per-element legend — too many). **Cyan quivers**
   show the Class C **ray** direction (perpendicular to skin-ring tangent,
   sign toward cap centroid) at skin vertices inside each cap's AABB, matching
   ``_resample_one_cap_interior_from_skin_rays``.

When skin junction splits raise strip/panel counts above subcomponent counts,
strip / panel subplots keep distinct colours per constituent instead of
collapsing to three kind colours. The element subplot colours per FE element.

Each figure suptitle includes **X** (web count), **Y** = ``structural_family``
(taxonomy spar-cap letter), **Z** (chord-normal vs flapwise), matching
``system_type_xyz_taxonomy.md`` (registry middle letter = **Y**).

By default writes one PNG for **every registered layout key** in
``SYSTEM_TYPE_KEYS``: the 38 multicell keys (**X** = 1..5, **Y** = A/B/C/D,
**Z** = CN or F, excluding **``1D-*``** — a single web cannot form a
continuous box spar) plus **``0A``** (X=0, Y=A: skin-only, no webs) and
**``0B``** (X=0, Y=B: skin + max-thickness cap band, cap SDF deferred so
currently skin-only too).  Total: 40 keys.

Also writes ``taxonomy_coverage.json`` in the same output directory (expected
**XYZ** set vs what was rendered, and ``excluded_invalid_xyz``).

Optional CLI: pass layout keys to render a subset, e.g.
``python -m blade_structure_utilities.fem_utilities.section_shell_model.examples.repro_section_centerline_mesh_v2 0A 0B 2D-F 2A-CN``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _all_layout_keys() -> tuple[str, ...]:
    """All registered layout keys: ``0A``, ``0B``, and all multicell ``XY-Z`` keys."""
    from blade_structure_utilities.geometry_utilities.section_geometry.core.system_layout import SYSTEM_TYPE_KEYS

    return SYSTEM_TYPE_KEYS


def _target_element_length_demo_m(chord_m: float) -> float:
    """~uniform physical spacing for MITC4 demo (metres), scaled from chord."""
    from blade_structure_utilities.mesh_utilities.section_mesh.strip_mesh import (
        TARGET_ELEMENT_LENGTH_FRAC,
        target_element_length_m_from_chord,
    )

    return target_element_length_m_from_chord(chord_m, frac=TARGET_ELEMENT_LENGTH_FRAC)


def _smoke_geo_export_vs_shell_inputs(
    strips_geo: tuple[Any, ...],
    shell_inputs: Any,
    layout_key: str,
    np: Any,
) -> None:
    """``build_shell_mesh_inputs`` may split skin and snap web/cap ends to the skin midline.

    Cap strips may additionally run
    :func:`~blade_structure_utilities.mesh_utilities.section_mesh.shell_inputs._resample_cap_interiors_from_skin_rays`,
    which replaces interior vertices (different ``N`` than ``n_cap_samples``) while
    keeping endpoints; those caps are only checked for endpoint agreement here.
    """
    geo_map = {(s.label, s.kind): s for s in strips_geo}
    shell_rows = list(shell_inputs.midlines)
    shell_map = {(m.label, m.kind): m for m in shell_rows}

    for key, s in geo_map.items():
        lab, kind = key
        if kind == "skin":
            segs = [
                m
                for m in shell_rows
                if m.kind == "skin" and str(m.label).startswith(f"{lab}:")
            ]
            if not segs:
                exact = [m for m in shell_rows if m.kind == "skin" and m.label == lab]
                assert exact and np.allclose(
                    s.midline_b, exact[0].midline_b, atol=1e-9, rtol=0.0
                ), f"[{layout_key}] missing or mismatched skin strip for {lab!r}"
            continue
        assert key in shell_map, f"[{layout_key}] missing shell midline for {key!r}"
        m = shell_map[key]
        ps = np.asarray(s.midline_b, dtype=float)
        pm = np.asarray(m.midline_b, dtype=float)
        # Webs: endpoints move onto the skin midline (~mm). Caps: ends can move
        # to the nearest snapped web foot within shell_inputs (up to tens of mm
        # under twist / real chord) while interiors may be ray-resampled.
        ep_tol = 0.025
        cap_ep_tol = 0.055
        if kind == "cap":
            assert ps.shape[1] == pm.shape[1] == 2, f"[{layout_key}] cap midline must be Nx2"
            assert float(np.linalg.norm(ps[0] - pm[0])) <= cap_ep_tol, (
                f"[{layout_key}] start endpoint drift for {key!r}"
            )
            assert float(np.linalg.norm(ps[-1] - pm[-1])) <= cap_ep_tol, (
                f"[{layout_key}] end endpoint drift for {key!r}"
            )
            continue
        assert ps.shape == pm.shape, f"[{layout_key}] shape mismatch for {key!r}"
        if ps.shape[0] > 2:
            assert np.allclose(ps[1:-1], pm[1:-1], atol=1e-9, rtol=0.0), (
                f"[{layout_key}] interior midline_b mismatch for {key!r}"
            )
        assert float(np.linalg.norm(ps[0] - pm[0])) <= ep_tol, (
            f"[{layout_key}] start endpoint drift for {key!r}"
        )
        assert float(np.linalg.norm(ps[-1] - pm[-1])) <= ep_tol, (
            f"[{layout_key}] end endpoint drift for {key!r}"
        )


def _run_single_layout(
    *,
    layout_key: str,
    airfoil,
    twist_rad: float,
    twist_deg: float,
    out_dir: Path,
) -> None:
    import numpy as np

    from blade_structure_utilities.geometry_utilities.section_contract.shell_midline_strip import ShellMidlineStrip
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (
        build_section_view,
        resolve_system_type,
    )
    from blade_structure_utilities.geometry_utilities.section_geometry.interface.shell_midline_export import (
        build_shell_midline_strips,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh.strip_mesh import (
        ENDPOINT_TOL_FRAC,
        TARGET_ELEMENT_LENGTH_FRAC,
        build_section_strip_mesh,
        target_element_length_m_from_chord,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh.shell_inputs import (
        ShellMeshInputs,
        build_shell_mesh_inputs,
    )

    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=twist_rad)

    # Smoke: geometry-export path produces the same strips as the adapter.
    strips_geo = build_shell_midline_strips(
        section, twist_rad=twist_rad, n_web_samples=20, n_cap_samples=80
    )
    shell_inputs = build_shell_mesh_inputs(
        section=section,
        twist_rad=twist_rad,
        layout_key=layout_key,
        n_web_samples=20,
        n_cap_samples=80,
    )

    _smoke_geo_export_vs_shell_inputs(strips_geo, shell_inputs, layout_key, np)
    assert isinstance(shell_inputs, ShellMeshInputs)
    assert all(isinstance(m, ShellMidlineStrip) for m in shell_inputs.midlines)

    chord_m = float(shell_inputs.chord_m)
    target_len_m = _target_element_length_demo_m(chord_m)

    mesh = build_section_strip_mesh(
        shell_inputs,
        target_element_length_frac=TARGET_ELEMENT_LENGTH_FRAC,
        endpoint_tol_frac=ENDPOINT_TOL_FRAC,
    )

    fig_path = out_dir / f"section_mesh_{layout_key}_i000.png"

    from blade_structure_utilities.fem_utilities.section_shell_model.vis import (
        save_section_centerline_mesh_figure,
        taxonomy_plot_caption,
    )

    save_section_centerline_mesh_figure(
        fig_path,
        mesh=mesh,
        section=section,
        shell_inputs=shell_inputs,
        layout=layout,
        layout_key=layout_key,
        airfoil=airfoil,
        twist_rad=twist_rad,
        twist_deg=twist_deg,
        target_element_length_m=target_len_m,
        station_tag="NACA63-415",
    )

    summ = mesh.summary()
    tax = taxonomy_plot_caption(layout_key, layout)
    print(f"[section_mesh_v2] wrote {fig_path}")
    print(
        f"[section_mesh_v2] {layout_key}: target_len={target_len_m:.4g} m -> "
        f"{summ['n_total_elements']} elements -- {tax}"
    )


def main() -> None:
    import sys

    import numpy as np

    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import resolve_system_type
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import AirfoilSDF

    chord_m = 1.655959896
    twist_deg = 34.110405083
    naca_series, naca_m, naca_p, naca_xx = 6, 63.0, 4.0, 15.0

    airfoil = AirfoilSDF.from_naca_series(
        naca_series,
        naca_m,
        naca_p,
        naca_xx,
        n_points=200,
        chord=chord_m,
        closed_te=True,
    )
    twist_rad = float(np.deg2rad(twist_deg))

    out_dir = Path(__file__).resolve().parent / "output" / "section_centerline_mesh_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = sys.argv[1:]
    if argv:
        layout_keys = [resolve_system_type(k.strip()).key for k in argv if k.strip()]
    else:
        layout_keys = list(_all_layout_keys())

    if not layout_keys:
        raise SystemExit("No layout keys to render.")

    rendered_png_names = {f"section_mesh_{k}_i000.png" for k in layout_keys}
    stale_pngs = sorted(
        p.name for p in out_dir.glob("section_mesh_*_i000.png") if p.name not in rendered_png_names
    )
    for name in stale_pngs:
        try:
            (out_dir / name).unlink()
        except Exception:
            pass

    # Expected full registry: 0A, 0B, plus X=1..5 x Y=A/B/C/D x Z=CN/F minus 1D-*.
    expected_keys = {"0A", "0B"} | {
        f"{x}{y}-{z}"
        for x in range(1, 6)
        for y in ("A", "B", "C", "D")
        for z in ("CN", "F")
        if not (x == 1 and y == "D")
    }
    rendered_set = set(layout_keys)
    coverage_manifest = {
        "rendered_keys": list(layout_keys),
        "missing_vs_full_registry": sorted(expected_keys - rendered_set),
        "extra_vs_full_registry": sorted(rendered_set - expected_keys),
        "excluded_invalid_keys": ["1D-CN", "1D-F"],
        "stale_pngs_removed": stale_pngs,
    }
    with open(out_dir / "taxonomy_coverage.json", "w", encoding="utf-8") as f:
        json.dump(coverage_manifest, f, indent=2)

    print(
        f"[section_mesh_v2] rendering {len(layout_keys)} layout(s): "
        f"{', '.join(layout_keys)}"
    )
    for layout_key in layout_keys:
        _run_single_layout(
            layout_key=layout_key,
            airfoil=airfoil,
            twist_rad=twist_rad,
            twist_deg=twist_deg,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
