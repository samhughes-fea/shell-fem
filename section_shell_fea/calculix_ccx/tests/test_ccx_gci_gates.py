"""Gate-profile tests for CCX membrane-only GCI (offline)."""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.convergence.grid_independence_study import (
    LayoutDriftSummary,
    evaluate_layout_gates,
)


def _summary(**kwargs) -> LayoutDriftSummary:
    base = dict(
        layout="2A-F",
        family="A",
        has_spar_caps=False,
        present_kinds=("skin", "web"),
        full_drift={
            "peak_abs_n12": {"skin": 1.1, "web": 1.2},
            "peak_mid_sig": {"skin": 9.0, "web": 9.0},
            "peak_hashin": {"skin": 9.0, "web": 9.0},
        },
        last_two_drift={
            "peak_abs_n12": {"skin": 1.05, "web": 1.1},
            "peak_mid_sig": {"skin": 9.0, "web": 9.0},
            "peak_hashin": {"skin": 9.0, "web": 9.0},
        },
        production_drift={
            "peak_abs_n12": {"skin": 1.1, "web": 1.2},
            "peak_mid_sig": {"skin": 9.0, "web": 9.0},
            "peak_hashin": {"skin": 9.0, "web": 9.0},
        },
        interior_last_two_drift={
            "peak_abs_n12": {"skin": 1.05, "web": 1.1},
            "peak_mid_sig": {"skin": 9.0, "web": 9.0},
            "peak_hashin": {"skin": 9.0, "web": 9.0},
        },
        gci_fine={
            "peak_abs_n12": {"skin": 0.02, "web": 0.03},
            "peak_mid_sig": {"skin": 0.5, "web": 0.5},
            "peak_hashin": {"skin": 0.5, "web": 0.5},
        },
        gci_ok={
            "peak_abs_n12": {"skin": True, "web": True},
            "peak_mid_sig": {"skin": True, "web": True},
            "peak_hashin": {"skin": True, "web": True},
        },
        gci_detail={
            "peak_abs_n12": {"skin": "ok", "web": "ok"},
            "peak_mid_sig": {"skin": "ok", "web": "ok"},
            "peak_hashin": {"skin": "ok", "web": "ok"},
        },
        governing_fi_full_drift=9.0,
        governing_fi_last_two_drift=9.0,
        governing_fi_production_drift=9.0,
        governing_fi_gci_fine=0.5,
        governing_fi_gci_ok=True,
        governing_fi_gci_detail="ok",
        all_mesh_ok=True,
        all_trust_ok=True,
    )
    base.update(kwargs)
    return LayoutDriftSummary(**base)


def test_ccx_gates_ignore_hashin_and_governing():
    fails = evaluate_layout_gates(_summary(), gates="ccx")
    assert fails == []


def test_ccx_gates_fail_on_nxy_gci():
    s = _summary(
        gci_fine={
            "peak_abs_n12": {"skin": 0.25, "web": 0.03},
            "peak_mid_sig": {"skin": 0.5, "web": 0.5},
            "peak_hashin": {"skin": 0.5, "web": 0.5},
        }
    )
    fails = evaluate_layout_gates(s, gates="ccx")
    assert any(f.metric == "peak_abs_n12_gci_fine" and f.kind == "skin" for f in fails)
    assert not any("hashin" in f.metric for f in fails)
    assert not any("governing" in f.metric for f in fails)


def test_study_membrane_only_skips_gov():
    fails = evaluate_layout_gates(_summary(), gates="study", membrane_only=True)
    assert fails == []
    assert not any("governing" in f.metric for f in fails)
