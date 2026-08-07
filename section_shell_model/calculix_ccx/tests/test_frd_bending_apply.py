"""Unit tests for FRD bending merge diagnostics (no live ccx required)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_model.calculix_ccx.results import (
    CcxFrdIpStress,
    apply_frd_bending_to_panel_resultants,
    bending_moments_from_frd_ips,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import (
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
        Nx=0.0, Ny=0.0, Nxy=0.0, Mx=0.0, My=0.0, Mxy=0.0,
        Qx=0.0, Qy=0.0, thickness_m=0.01, panel_index=0, station_index=0,
        provenance={
            k: FieldProvenance(ProvenanceKind.RESERVED, "needs .frd through-thickness")
            for k in ("Mx", "My", "Mxy")
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
    assert res.provenance["Mx"].kind is ProvenanceKind.CALCULIX_CCX
    assert res.provenance["My"].kind is ProvenanceKind.CALCULIX_CCX
    assert res.provenance["Mxy"].kind is ProvenanceKind.CALCULIX_CCX
    assert abs(res.Mx) + abs(res.My) + abs(res.Mxy) > 0.0


def test_apply_frd_bending_missing_ip_keeps_reserved_with_reason():
    mesh = _mesh_one_elem()
    res = ShellPanelResultants(
        Nx=0.0, Ny=0.0, Nxy=0.0, Mx=0.0, My=0.0, Mxy=0.0,
        thickness_m=0.01, panel_index=0,
        provenance={
            k: FieldProvenance(ProvenanceKind.RESERVED, "needs .frd through-thickness")
            for k in ("Mx", "My", "Mxy")
        },
    )
    stats = apply_frd_bending_to_panel_resultants([[res]], mesh, {})
    assert stats["n_missing_ip"] == 1
    assert stats["n_applied"] == 0
    assert res.provenance["Mx"].kind is ProvenanceKind.RESERVED
    assert "no IP" in res.provenance["Mx"].note
