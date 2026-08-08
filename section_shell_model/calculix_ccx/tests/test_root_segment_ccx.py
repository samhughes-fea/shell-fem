"""Offline deck/mesh tests + optional live ccx solve for the root-segment diagnostic."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_model.calculix_ccx.results import (
    CcxElementStress,
    map_root_segment_stresses_to_resultants,
    parse_ccx_dat_node_reactions,
)
from blade_structure_utilities.fem_utilities.section_shell_model.calculix_ccx.root_segment_export import (
    outboard_cloads_from_resultants,
    run_root_segment_ccx,
    write_root_segment_inp,
)
from blade_structure_utilities.fem_utilities.section_shell_model.calculix_ccx.root_segment_mesh import (
    build_ccx_root_segment_mesh,
    estimate_root_diameter_m,
)
from blade_structure_utilities.material_utilities.adapters.default_laminates import (
    WallLaminate,
)

_HAS_CCX = shutil.which("ccx") is not None


class _Panel:
    def __init__(self, label, nodes):
        self.label = label
        self.nodes = np.asarray(nodes, dtype=float)
        ds = np.maximum(
            np.hypot(np.diff(self.nodes[:, 0]), np.diff(self.nodes[:, 1])), 1e-30
        )
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        self.lam = WallLaminate(E=70e9, t=0.01, nu=0.33, name="iso", n_plies=1)
        self.cell_id = 0
        self.end_boom = None


def _panels():
    return [
        _Panel("skin:lower", [[0.0, 0.0], [1.0, 0.0]]),
        _Panel("web:1", [[1.0, 0.0], [1.0, 0.5]]),
    ]


def test_estimate_root_diameter():
    d = estimate_root_diameter_m(_panels())
    assert d == pytest.approx(1.0)


def test_root_segment_mesh_row_counts_and_nsets():
    panels = _panels()
    mesh = build_ccx_root_segment_mesh(
        panels, segment_length_m=2.5, n_span_rows=4, n_elements_per_panel=2
    )
    assert mesh.n_span_rows == 4
    assert mesh.span_coords.size == 5
    assert abs(mesh.span_coords[0]) < 1e-15
    assert mesh.span_coords[-1] == pytest.approx(2.5)
    # 2 panels × 2 contour elems × 4 span rows
    assert len(mesh.elements) == 2 * 2 * 4
    assert all(e.span_row in range(4) for e in mesh.elements)
    assert len(mesh.nsets["ROOT_RING"]) >= 3
    assert len(mesh.nsets["OUTBOARD_RING"]) >= 3
    for nid in mesh.nsets["ROOT_RING"]:
        assert mesh.nodes[nid][0] == pytest.approx(0.0)
    for nid in mesh.nsets["OUTBOARD_RING"]:
        assert mesh.nodes[nid][0] == pytest.approx(2.5)


def test_root_segment_loft_interpolates_yz():
    """Two guide stations with different web height → outboard YZ differs from root."""
    root = _panels()
    out = [
        _Panel("skin:lower", [[0.0, 0.0], [1.0, 0.0]]),
        _Panel("web:1", [[1.0, 0.0], [1.0, 1.0]]),
    ]
    mesh = build_ccx_root_segment_mesh(
        root,
        segment_length_m=1.0,
        n_span_rows=2,
        n_elements_per_panel=2,
        station_panels=[root, out],
    )
    web_out_z = [
        mesh.nodes[nid][2]
        for nid in mesh.nsets["OUTBOARD_RING"]
        if abs(mesh.nodes[nid][1] - 1.0) < 1e-9
    ]
    assert web_out_z
    assert max(web_out_z) == pytest.approx(1.0, rel=1e-6)


def test_root_segment_deck_has_clamp_outboard_cload_and_s4r():
    panels = _panels()
    mesh = build_ccx_root_segment_mesh(
        panels, segment_length_m=2.0, n_span_rows=3, n_elements_per_panel=2
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = write_root_segment_inp(
            mesh, panels, Path(tmp) / "root.inp", My=1.0e5, Vz=1.0e4, N=1.0e5
        )
        text = path.read_text(encoding="utf-8")
    assert "*ELEMENT, TYPE=S4R" in text
    assert "ROOT_RING, 1, 6" in text
    assert "*RIGID BODY" not in text
    assert "*COUPLING" not in text
    assert "*CLOAD" in text
    assert "*NODE PRINT, NSET=ROOT_RING" in text
    assert text.count("*STEP") == 1
    # Helper REF/ROT nodes are not written into the deck
    assert f"\n{mesh.ref_node_id}," not in text
    assert f"\n{mesh.rot_node_id}," not in text


def test_outboard_cloads_sum_to_applied_forces():
    panels = _panels()
    mesh = build_ccx_root_segment_mesh(
        panels, segment_length_m=2.0, n_span_rows=3, n_elements_per_panel=2
    )
    cloads = outboard_cloads_from_resultants(
        mesh, panels, N=1.0e5, Vy=2.0e4, Vz=-3.0e4, My=5.0e4, T=1.0e3
    )
    sx = sum(v for _n, d, v in cloads if d == 1)
    sy = sum(v for _n, d, v in cloads if d == 2)
    sz = sum(v for _n, d, v in cloads if d == 3)
    assert sx == pytest.approx(1.0e5, rel=1e-9)
    assert sy == pytest.approx(2.0e4, rel=1e-9)
    assert sz == pytest.approx(-3.0e4, rel=1e-9)
    # Moments / torsion preserve net force on Fx (My/Mz couples sum to 0)
    # Already covered by sx ≈ N.


def test_map_root_segment_stresses_decay_profile():
    panels = _panels()
    mesh = build_ccx_root_segment_mesh(
        panels, segment_length_m=2.0, n_span_rows=2, n_elements_per_panel=2
    )
    stresses = []
    for e in mesh.elements:
        sxx = 100.0 * (1.0 + e.span_row)
        stresses.append(
            CcxElementStress(
                eid=e.eid, sxx=sxx, syy=0.0, szz=0.0, sxy=0.0, sxz=0.0, syz=0.0
            )
        )
    mapped = map_root_segment_stresses_to_resultants(mesh, panels, stresses)
    assert len(mapped.root_row) == 2
    assert len(mapped.decay_profile) == 2
    assert mapped.decay_profile[1]["max_abs_Nx"] > mapped.decay_profile[0]["max_abs_Nx"]
    assert abs(mapped.by_panel_row[0][1][0].Nx) > abs(mapped.by_panel_row[0][0][0].Nx)


def test_parse_ccx_dat_node_reactions_nset_hint():
    dat = """
 forces (fx,fy,fz) for set ROOT_RING and time  0.1000000E+01

         1  1.0e+03  0.0  0.0
         2  2.0e+03  0.0  0.0

 forces (fx,fy,fz) for set OUTBOARD_RING and time  0.1000000E+01

        99  -3.0e+03  0.0  0.0
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.dat"
        path.write_text(dat, encoding="utf-8")
        root = parse_ccx_dat_node_reactions(path, nset_hint="ROOT_RING")
        outb = parse_ccx_dat_node_reactions(path, nset_hint="OUTBOARD_RING")
    assert root[1][0] == pytest.approx(1.0e3)
    assert root[2][0] == pytest.approx(2.0e3)
    assert outb[99][0] == pytest.approx(-3.0e3)


@pytest.mark.ccx
@pytest.mark.skipif(not _HAS_CCX, reason="ccx not on PATH")
def test_live_root_segment_axial_equilibrium():
    """Applied N should be balanced by root ΣRF_x (opposite sign)."""
    panels = _panels()
    N = 1.0e5
    with tempfile.TemporaryDirectory() as tmp:
        _mesh, mapped, diag, _inp = run_root_segment_ccx(
            panels,
            work_dir=tmp,
            segment_length_m=2.0,
            n_span_rows=4,
            n_elements_per_panel=2,
            N=N,
            job_name="root_ax",
        )
    rx, _ry, _rz = diag["root_rf_sum"]
    assert diag["n_root_rf_nodes"] > 0
    assert abs(rx + N) / N < 0.15
    assert mapped.decay_profile
    assert any(abs(r.Nx) > 0.0 for panel in mapped.root_row for r in panel)
