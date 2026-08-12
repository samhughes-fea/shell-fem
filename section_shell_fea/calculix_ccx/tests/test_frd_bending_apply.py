"""Unit tests for FRD bending merge diagnostics (no live ccx required)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.results import (
    CcxFrdIpStress,
    apply_frd_bending_to_panel_resultants,
    bending_moments_from_frd_ips,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    FieldProvenance,
    ProvenanceKind,
    ShellPanelResultants,
)


def _mesh_one_elem():
    elem = SimpleNamespace(
        eid=1,
        panel_index=0,
        local_element_index=0,
        node_ids=(1, 2, 3, 4),
    )
    pm = SimpleNamespace(panel_label="skin:lower", panel_index=0)
    mesh = SimpleNamespace(
        panel_maps=[pm],
        elements=[elem],
        nodes={
            1: (0.0, 0.0, 0.0),
            2: (0.0, 1.0, 0.0),
            3: (0.1, 1.0, 0.0),
            4: (0.1, 0.0, 0.0),
        },
    )
    return mesh


def test_bending_moments_from_frd_ips_nontrivial_spread():
    ips = [
        CcxFrdIpStress(1, 1, 1e6, 0.0, 0.0, 0.0, 0.0, 0.0),
        CcxFrdIpStress(1, 2, -1e6, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    mx, my, mxy = bending_moments_from_frd_ips(ips, thickness=0.01, ty=1.0, tz=0.0)
    assert mx > 0.0
    assert my == 0.0
    assert mxy == 0.0


def test_apply_frd_bending_sets_all_components_and_stats():
    mesh = _mesh_one_elem()
    res = ShellPanelResultants(
        N11=0.0, N22=0.0, N12=0.0, M11=0.0, M22=0.0, M12=0.0,
        Q1=0.0, Q2=0.0, thickness_m=0.01, panel_index=0, station_index=0,
        provenance={
            k: FieldProvenance(ProvenanceKind.RESERVED, "needs .frd through-thickness")
            for k in ("M11", "M22", "M12")
        },
    )
    frd = {
        1: [
            CcxFrdIpStress(1, 1, 2e6, 1e5, 0.0, 5e4, 0.0, 0.0),
            CcxFrdIpStress(1, 2, -2e6, -1e5, 0.0, -5e4, 0.0, 0.0),
        ]
    }
    stats = apply_frd_bending_to_panel_resultants([[res]], mesh, frd)
    assert stats["n_elements"] == 1
    assert stats["n_applied"] == 1
    assert stats["n_missing_ip"] == 0
    assert res.provenance["M11"].kind is ProvenanceKind.CALCULIX_CCX
    assert res.provenance["M22"].kind is ProvenanceKind.CALCULIX_CCX
    assert res.provenance["M12"].kind is ProvenanceKind.CALCULIX_CCX
    assert abs(res.M11) + abs(res.M22) + abs(res.M12) > 0.0


def test_apply_frd_bending_missing_ip_keeps_reserved_with_reason():
    mesh = _mesh_one_elem()
    res = ShellPanelResultants(
        N11=0.0, N22=0.0, N12=0.0, M11=0.0, M22=0.0, M12=0.0,
        thickness_m=0.01, panel_index=0,
        provenance={
            k: FieldProvenance(ProvenanceKind.RESERVED, "needs .frd through-thickness")
            for k in ("M11", "M22", "M12")
        },
    )
    stats = apply_frd_bending_to_panel_resultants([[res]], mesh, {})
    assert stats["n_missing_ip"] == 1
    assert stats["n_applied"] == 0
    assert res.provenance["M11"].kind is ProvenanceKind.RESERVED
    assert "no IP" in res.provenance["M11"].note
