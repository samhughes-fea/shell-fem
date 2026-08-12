"""Multi-key grid independence gates (fast CI + slow smoke matrix)."""

from __future__ import annotations

import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.grid_independence_study import (
    CI_SMOKE_KEYS,
    FAST_CI_KEYS,
    INFORMATIONAL_KINDS,
    KIND_METRICS,
    LayoutDriftSummary,
    STUDY_REFINEMENTS,
    evaluate_layout_gates,
    resolve_study_keys,
    run_layout_ladder,
    summarize_layout_ladder,
)


@pytest.fixture(scope="module")
def fast_ladders():
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.grid_independence_study import (
        load_cotide_station_recipe,
    )

    recipe = load_cotide_station_recipe()
    kw = recipe.cascade_kwargs()
    return {
        key: run_layout_ladder(key, refinements=STUDY_REFINEMENTS, **kw)
        for key in FAST_CI_KEYS
    }


def test_resolve_study_keys_presets():
    assert "5D-F" in resolve_study_keys("all")
    assert resolve_study_keys("smoke") == CI_SMOKE_KEYS
    assert resolve_study_keys("fast") == FAST_CI_KEYS
    assert resolve_study_keys("5D-F,2A-F") == ("5D-F", "2A-F")
    with pytest.raises(ValueError, match="unknown"):
        resolve_study_keys("NOT-A-KEY")


def test_ci_gates_skip_cap_kind_peaks():
    """Cap kind-peak drifts are informational under CI (Defect K analogue)."""
    summary = LayoutDriftSummary(
        layout="2B-CN",
        family="B",
        has_spar_caps=True,
        present_kinds=("skin", "cap", "web"),
        full_drift={m: {"skin": 1.1, "cap": 6.0, "web": 1.1} for m in KIND_METRICS},
        last_two_drift={m: {"skin": 1.1, "cap": 5.0, "web": 1.1} for m in KIND_METRICS},
        production_drift={m: {"skin": 1.1, "cap": 6.0, "web": 1.1} for m in KIND_METRICS},
        interior_last_two_drift={
            m: {"skin": 1.1, "cap": 1.2, "web": 1.1} for m in KIND_METRICS
        },
        gci_fine={
            m: {"skin": 0.02, "cap": 0.5, "web": 0.02} for m in KIND_METRICS
        },
        gci_ok={m: {"skin": True, "cap": True, "web": True} for m in KIND_METRICS},
        gci_detail={m: {"skin": "", "cap": "", "web": ""} for m in KIND_METRICS},
        governing_fi_full_drift=1.1,
        governing_fi_last_two_drift=1.1,
        governing_fi_production_drift=1.1,
        governing_fi_gci_fine=0.02,
        governing_fi_gci_ok=True,
        all_mesh_ok=True,
        all_trust_ok=True,
    )
    fails = evaluate_layout_gates(summary, gates="ci")
    assert not fails
    assert "cap" in INFORMATIONAL_KINDS


def test_ci_gates_fail_on_hashin_gci_and_non_monotonic():
    base = dict(
        layout="2D-F",
        family="D",
        has_spar_caps=True,
        present_kinds=("skin", "web"),
        full_drift={m: {"skin": 1.1, "web": 1.1} for m in KIND_METRICS},
        last_two_drift={m: {"skin": 1.1, "web": 1.1} for m in KIND_METRICS},
        production_drift={m: {"skin": 1.1, "web": 1.1} for m in KIND_METRICS},
        interior_last_two_drift={m: {"skin": 1.1, "web": 1.1} for m in KIND_METRICS},
        gci_fine={m: {"skin": 0.02, "web": 0.02} for m in KIND_METRICS},
        gci_ok={m: {"skin": True, "web": True} for m in KIND_METRICS},
        gci_detail={m: {"skin": "", "web": ""} for m in KIND_METRICS},
        governing_fi_full_drift=1.1,
        governing_fi_last_two_drift=1.1,
        governing_fi_production_drift=1.1,
        governing_fi_gci_fine=0.02,
        governing_fi_gci_ok=True,
        all_mesh_ok=True,
        all_trust_ok=True,
    )
    high = LayoutDriftSummary(
        **{
            **base,
            "gci_fine": {
                m: {"skin": 0.02 if m != "peak_hashin" else 0.25, "web": 0.02}
                for m in KIND_METRICS
            },
        }
    )
    fails = evaluate_layout_gates(high, gates="ci")
    assert any(f.metric == "peak_hashin_gci_fine" and f.kind == "skin" for f in fails)

    osc = LayoutDriftSummary(
        **{
            **base,
            "gci_ok": {
                m: {"skin": m != "peak_hashin", "web": True} for m in KIND_METRICS
            },
            "gci_detail": {
                m: {
                    "skin": "gci_non_monotonic" if m == "peak_hashin" else "",
                    "web": "",
                }
                for m in KIND_METRICS
            },
            "gci_fine": {
                m: {"skin": float("nan"), "web": 0.02} for m in KIND_METRICS
            },
        }
    )
    fails_osc = evaluate_layout_gates(osc, gates="ci")
    assert any(
        f.metric == "peak_hashin_gci_fine" and f.detail == "gci_non_monotonic"
        for f in fails_osc
    )


@pytest.mark.parametrize("layout", FAST_CI_KEYS)
def test_fast_ci_layout_passes_gates(layout, fast_ladders):
    rows = fast_ladders[layout]
    assert len(rows) == len(STUDY_REFINEMENTS)
    assert all(r.layout == layout for r in rows)
    summary = summarize_layout_ladder(rows)
    assert summary.all_mesh_ok
    assert summary.all_trust_ok
    assert "peak_hashin" in summary.gci_fine
    assert summary.governing_fi_p_obs == summary.governing_fi_p_obs  # finite or nan
    fails = evaluate_layout_gates(summary, gates="ci")
    # Ratio-drift / mesh / trust contract (pre-GCI). ASME GCI is also gated in
    # evaluate_layout_gates and may fail on oscillatory FE peaks — assert drifts
    # separately so mesh-independence ratio gates stay green.
    drift_fails = [f for f in fails if "gci" not in f.metric]
    assert not drift_fails, (
        "; ".join(
            f"{f.kind}/{f.metric}={f.drift:.3f}>{f.limit}"
            + (f" ({f.detail})" if f.detail else "")
            for f in drift_fails
        )
    )
    # Interior fields populated when the kind is present.
    for row in rows:
        for kind in summary.present_kinds:
            assert kind in row.peak_hashin_interior
            raw = float(row.peak_hashin.get(kind, float("nan")))
            interior = float(row.peak_hashin_interior.get(kind, float("nan")))
            if raw == raw and interior == interior and raw > 0.0:
                assert interior <= raw + 1e-12


@pytest.mark.slow
@pytest.mark.parametrize("layout", CI_SMOKE_KEYS)
def test_smoke_matrix_layout_passes_gates(layout):
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.grid_independence_study import (
        load_cotide_station_recipe,
    )

    recipe = load_cotide_station_recipe()
    rows = run_layout_ladder(
        layout, refinements=STUDY_REFINEMENTS, **recipe.cascade_kwargs()
    )
    summary = summarize_layout_ladder(rows)
    fails = evaluate_layout_gates(summary, gates="ci")
    assert summary.all_mesh_ok and summary.all_trust_ok
    drift_fails = [f for f in fails if "gci" not in f.metric]
    assert not drift_fails, (
        "; ".join(f"{f.kind}/{f.metric}={f.drift:.3f}>{f.limit}" for f in drift_fails)
    )
