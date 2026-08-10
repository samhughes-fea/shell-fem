"""Unit tests for CalculiX ``.frd`` through-thickness stress parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_model.calculix_ccx.results import (
    _frd_to_mesh_eid,
    _parse_frd_minus1_line,
    bending_moments_from_frd_ips,
    parse_ccx_frd_3c_node_index,
    parse_ccx_frd_stress_blocks,
    superpose_frd_stress_blocks,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ccx_mini.frd"
_PACKED = Path(__file__).resolve().parent / "fixtures" / "ccx_packed.frd"
_3C_MINI = Path(__file__).resolve().parent / "fixtures" / "ccx_3c_mini.frd"


def test_parse_ccx_frd_stress_blocks_mini_fixture():
    blocks = parse_ccx_frd_stress_blocks(_FIXTURE)
    assert len(blocks) == 1
    rows = blocks[0]
    assert len(rows) == 4
    eids = {r.eid for r in rows}
    assert eids == {10, 20}
    r101 = next(r for r in rows if r.eid == 10 and r.ip == 1)
    assert r101.sxx == pytest.approx(1.0e6)
    r201 = next(r for r in rows if r.eid == 20 and r.ip == 1)
    assert r201.sxx == pytest.approx(1.6e6)


def test_parse_frd_minus1_line_packed_ccx223():
    """ccx 2.23 packed ``-1`` row: ``i10`` internal id + ip digit fused with first float."""
    line = (
        " -1        894.50000E+0102.52158E+001-4.65744E-0011.30288E-003"
        "-5.55442E-0015.69420E-003"
    )
    parsed = _parse_frd_minus1_line(line)
    assert parsed is not None
    internal_eid, ip, vals = parsed
    assert internal_eid == 89
    assert ip == 4
    assert vals[0] == pytest.approx(4.5e10)
    assert vals[1] == pytest.approx(2.52158e1)
    assert vals[2] == pytest.approx(-4.65744e-1)


def test_parse_ccx_frd_stress_blocks_packed_fixture():
    blocks = parse_ccx_frd_stress_blocks(_PACKED)
    assert len(blocks) == 1
    rows = blocks[0]
    assert len(rows) == 1
    r = rows[0]
    assert r.eid == 89
    assert r.ip == 4
    assert r.sxx == pytest.approx(4.5e10)


def test_frd_to_mesh_eid_hybrid_layout():
    base = 88
    assert _frd_to_mesh_eid(89, element_base=base, n_mesh_elements=44) == 1
    assert _frd_to_mesh_eid(133, element_base=base, n_mesh_elements=44) == 13
    assert _frd_to_mesh_eid(500, element_base=base, n_mesh_elements=44) is None


def test_parse_ccx_frd_3c_node_index_mini_fixture():
    index = parse_ccx_frd_3c_node_index(_3C_MINI)
    assert index is not None
    assert index.user_nodes[1] == frozenset({89, 91, 103, 105})
    assert index.user_nodes[2] == frozenset({257, 259, 266, 268})
    assert 89 in index.node_to_users
    assert set(index.node_to_users[89]) == {1}
    assert set(index.node_to_users[257]) == {2}


def test_frd_to_mesh_eid_uses_3c_node_index():
    index = parse_ccx_frd_3c_node_index(_3C_MINI)
    assert index is not None
    base = 88
    # Node 257 belongs to user element 2 only; must not divmod-map to 25.
    assert _frd_to_mesh_eid(257, element_base=base, n_mesh_elements=44, frd_index=index) == 2
    assert _frd_to_mesh_eid(89, element_base=base, n_mesh_elements=44, frd_index=index) == 1


def test_superpose_frd_stress_blocks_linear():
    blocks = parse_ccx_frd_stress_blocks(_FIXTURE)
    q = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    by_eid = superpose_frd_stress_blocks(blocks, q)
    assert by_eid[10][0].sxx == pytest.approx(2.0e6)


def test_bending_moments_from_frd_ips_flat_plate():
    from blade_structure_utilities.fem_utilities.section_shell_model.calculix_ccx.results import CcxFrdIpStress

    rows = [
        CcxFrdIpStress(1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        CcxFrdIpStress(1, 4, 120e6, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    t = 0.01
    mx, my, mxy = bending_moments_from_frd_ips(rows, thickness=t, ty=1.0, tz=0.0)
    assert mx == pytest.approx(120e6 * t * t / 12.0)
    assert my == pytest.approx(0.0)
    assert mxy == pytest.approx(0.0)


def test_bending_moments_from_frd_ips_contour_my():
    from blade_structure_utilities.fem_utilities.section_shell_model.calculix_ccx.results import CcxFrdIpStress

    rows = [
        CcxFrdIpStress(1, 1, 0.0, 10e6, 0.0, 0.0, 0.0, 0.0),
        CcxFrdIpStress(1, 4, 0.0, 50e6, 0.0, 0.0, 0.0, 0.0),
    ]
    t = 0.01
    mx, my, mxy = bending_moments_from_frd_ips(rows, thickness=t, ty=1.0, tz=0.0)
    assert mx == pytest.approx(0.0)
    assert my == pytest.approx(40e6 * t * t / 12.0)
    assert mxy == pytest.approx(0.0)
