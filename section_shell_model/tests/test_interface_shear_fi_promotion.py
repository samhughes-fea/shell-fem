"""Golden / parity gates for FSDT interface-shear interlaminar FI promotion."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.section_materials.laminate import LaminateDefinition
from blade_structure_utilities.section_materials.ply import OrthotropicPly
from blade_structure_utilities.section_shell_model.processing.static.results.api.fi_recovery import (
    compute_quaternary_fi_from_panel_rows,
    interlaminar_fi_from_row,
    interlaminar_fi_from_row_fsdt,
)
from blade_structure_utilities.section_shell_model.pre_processing.containers.types import (
    ShellPanelResultants,
    ShellTheory,
)
from blade_structure_utilities.section_shell_model.processing.static.results.compute_tertiary import (
    CcxIpShearRecoveryOperator,
    FSDTInterfaceShearRecoveryOperator,
    HSDTQxShearRecoveryOperator,
    resolve_shear_recovery_operator,
)


def _ply(t_ply: float = 1e-3) -> OrthotropicPly:
    return OrthotropicPly(
        name="gfrp",
        E1=30e9,
        E2=8e9,
        G12=3e9,
        nu12=0.3,
        rho=1.5e3,
        t_ply=t_ply,
        Xt=500e6,
        Xc=300e6,
        Yt=50e6,
        Yc=200e6,
        S12=40e6,
        Zt=1e6,
        S13=6e6,
        S23=6e6,
    )


def _row(*, qx: float = 100.0, qy: float = 50.0, t: float = 3e-3) -> ShellPanelResultants:
    return ShellPanelResultants(
        Nx=1e3, Ny=0.0, Nxy=0.0, Mx=0.0, My=0.0, Mxy=0.0,
        Qx=qx, Qy=qy, thickness_m=t, panel_index=0, station_index=0,
    )


def test_uniform_multi_ply_interface_fi_matches_scalar_peak():
    """Homogeneous 2-ply: midplane interface recovers peak 1.5 |Q|/t (= scalar)."""
    p = _ply(1e-3)
    # Two plies place an interface at z=0 where the parabola peaks at 1.5 Q/h.
    lam = LaminateDefinition(plies=[(p, 0.0), (p, 0.0)])
    row = _row(t=2e-3)
    s_il = 6e6
    fi_iface, nr, mode = interlaminar_fi_from_row_fsdt(row, lam, s_il=s_il)
    fi_scalar, nr_s = interlaminar_fi_from_row(row, s_il=s_il, peak_factor=1.5)
    assert not nr and not nr_s
    assert mode == "interface"
    assert fi_iface == pytest.approx(fi_scalar, rel=1e-9, abs=1e-12)


def test_single_ply_uses_scalar_fallback():
    lam = LaminateDefinition(plies=[(_ply(2e-3), 0.0)])
    row = _row(t=2e-3)
    fi, nr, mode = interlaminar_fi_from_row_fsdt(row, lam, s_il=6e6)
    fi_scalar, _ = interlaminar_fi_from_row(row, s_il=6e6, peak_factor=1.5)
    assert not nr
    assert mode == "scalar"
    assert fi == pytest.approx(fi_scalar)


def test_untrusted_q_returns_untrusted_mode():
    from blade_structure_utilities.section_shell_model.pre_processing.containers.types import (
        FieldProvenance,
        ProvenanceKind,
    )

    row = _row()
    row.provenance["Qx"] = FieldProvenance(ProvenanceKind.RESERVED, "missing Q")
    lam = LaminateDefinition(plies=[(_ply(), 0.0), (_ply(), 90.0)])
    fi, nr, mode = interlaminar_fi_from_row_fsdt(row, lam, s_il=6e6)
    assert fi == 0.0
    assert nr
    assert mode == "untrusted"


def test_four_tier_bridge_interlaminar_matches_row_fsdt():
    p = _ply(1e-3)
    lam = LaminateDefinition(plies=[(p, 0.0), (p, 0.0)])
    rows = [_row(qx=200.0, qy=0.0, t=2e-3)]
    bundle_fi = compute_quaternary_fi_from_panel_rows([[rows[0]]], [lam])
    fi_q = float(bundle_fi.quaternary.hashin_fi.interlaminar[0])
    fi_row, _nr, mode = interlaminar_fi_from_row_fsdt(rows[0], lam, s_il=6e6)
    assert mode == "interface"
    assert fi_q == pytest.approx(fi_row, rel=1e-9, abs=1e-12)


def test_resolve_shear_recovery_operator_theory_protocol():
    assert isinstance(resolve_shear_recovery_operator(ShellTheory.FSDT), FSDTInterfaceShearRecoveryOperator)
    assert isinstance(resolve_shear_recovery_operator(ShellTheory.CALCULIX_CCX), CcxIpShearRecoveryOperator)
    assert isinstance(resolve_shear_recovery_operator("HSDT"), HSDTQxShearRecoveryOperator)


def test_heterogeneous_g_interface_can_differ_from_naive_scalar():
    soft = _ply(1e-3)
    soft.G13 = 1e9
    soft.G23 = 1e9
    stiff = _ply(1e-3)
    stiff.G13 = 10e9
    stiff.G23 = 10e9
    lam = LaminateDefinition(plies=[(soft, 0.0), (stiff, 0.0), (soft, 0.0)])
    row = _row(qx=300.0, qy=0.0, t=3e-3)
    fi_iface, nr, mode = interlaminar_fi_from_row_fsdt(row, lam, s_il=6e6)
    fi_scalar, _ = interlaminar_fi_from_row(row, s_il=6e6, peak_factor=1.5)
    assert not nr and mode == "interface"
    # Stiffness weighting redistributes peak away from the uniform 1.5 Q/t value.
    assert abs(fi_iface - fi_scalar) > 1e-6 * max(fi_scalar, 1e-12)
