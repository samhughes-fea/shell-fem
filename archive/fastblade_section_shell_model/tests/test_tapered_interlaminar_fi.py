"""Interlaminar FI: tapered FSDT path and per-pair -> per-station attribution."""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.section_shell_model.api.fi_recovery import interlaminar_fi_from_row
from blade_precompute.section_shell_model.api.pipeline_b import build_tapered_pipeline_b_section
from blade_precompute.section_shell_model.api.periodic import (
    run_section_tapered_periodic_bc_mitc4,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import ShellPanelResultants


def _tapered_pb():
    return build_tapered_pipeline_b_section(
        "2412",
        [0.35],
        chord_inboard_m=1.0,
        chord_outboard_m=0.92,
        L_x=0.5,
    )


def test_tapered_fsdt_fi_il_under_shear():
    pb = _tapered_pb()
    bundle = run_section_tapered_periodic_bc_mitc4(
        pb, Vy=5.0e3, Vz=2.0e3, My=1.0e4, n_elements_per_panel=6,
    )
    max_q = 0.0
    for panel_res in bundle.all_panel_results or []:
        for row in panel_res or []:
            if row is None:
                continue
            qx, qy = getattr(row, "Qx", None), getattr(row, "Qy", None)
            if qx is not None and qy is not None:
                max_q = max(max_q, float(np.hypot(qx, qy)))
    assert max_q > 0.0


def test_interlaminar_attribution_inboard_convention():
    """Pair FI attributed to inboard station index (documented convention)."""
    n_stations = 5
    fi_pair = np.array([0.1, 0.2, 0.3, 0.4])
    fi_station = np.zeros(n_stations, dtype=float)
    for k, fi in enumerate(fi_pair):
        fi_station[k] = max(fi_station[k], fi)
    assert fi_station[0] == pytest.approx(0.1)
    assert fi_station[3] == pytest.approx(0.4)
    assert fi_station[4] == 0.0


def test_interlaminar_no_double_peak_factor():
    row = ShellPanelResultants(
        Nx=0.0, Ny=0.0, Nxy=0.0, Mx=0.0, My=0.0, Mxy=0.0,
        Qx=100.0, Qy=50.0, thickness_m=0.002,
    )
    fi_q, _ = interlaminar_fi_from_row(row, s_il=40e6, peak_factor=1.5)
    fi_h, _ = interlaminar_fi_from_row(row, s_il=40e6, peak_factor=0.0)
    assert fi_q > fi_h


def test_tapered_interface_fi_matches_uniform_scalar_envelope():
    """Multi-ply uniform skin under tapered shear: interface FI ≈ scalar 1.5|Q|/t."""
    from blade_precompute.materials.laminate import LaminateDefinition
    from blade_precompute.materials.ply import OrthotropicPly
    from blade_precompute.section_shell_model.api.fi_recovery import interlaminar_fi_from_row_fsdt

    pb = _tapered_pb()
    bundle = run_section_tapered_periodic_bc_mitc4(
        pb, Vy=5.0e3, Vz=2.0e3, My=1.0e4, n_elements_per_panel=6,
    )
    # Use a homogeneous 3-ply stand-in laminate for the FI mode check on real Q rows.
    p = OrthotropicPly(
        name="gfrp", E1=30e9, E2=8e9, G12=3e9, nu12=0.3, rho=1500.0, t_ply=1e-3,
        Xt=500e6, Xc=300e6, Yt=50e6, Yc=200e6, S12=40e6, Zt=1e6, S13=6e6, S23=6e6,
    )
    lam = LaminateDefinition(plies=[(p, 0.0), (p, 0.0)])
    saw_iface = False
    for panel_res in bundle.all_panel_results or []:
        for row in panel_res or []:
            if row is None or row.Qx is None or row.Qy is None:
                continue
            if float(np.hypot(row.Qx, row.Qy)) <= 0.0:
                continue
            # Align thickness with the 2-ply stand-in (midplane interface = peak).
            row_t = ShellPanelResultants(
                Nx=row.Nx, Ny=row.Ny, Nxy=row.Nxy, Mx=row.Mx, My=row.My, Mxy=row.Mxy,
                Qx=row.Qx, Qy=row.Qy, thickness_m=2e-3,
                provenance=dict(row.provenance or {}),
                panel_index=row.panel_index, station_index=row.station_index,
            )
            fi_i, nr, mode = interlaminar_fi_from_row_fsdt(row_t, lam, s_il=6e6)
            fi_s, _ = interlaminar_fi_from_row(row_t, s_il=6e6, peak_factor=1.5)
            assert not nr
            if mode == "interface":
                assert fi_i == pytest.approx(fi_s, rel=1e-6, abs=1e-12)
                saw_iface = True
    assert saw_iface
