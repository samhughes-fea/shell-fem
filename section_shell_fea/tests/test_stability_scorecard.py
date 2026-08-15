"""Unit tests for stability scorecard magnitude / Roache split + watch tagging."""

from __future__ import annotations

import csv
from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.grid_independence_study import (
    CI_WATCH_KEYS,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.stability_scorecard import (
    build_stability_scorecard,
    classify_gci_fail,
    write_stability_scorecard,
)


def test_classify_magnitude_vs_roache():
    assert (
        classify_gci_fail(
            gci_fine=0.18, gci_ok=False, detail="", limit=0.10
        )
        == "magnitude"
    )
    assert (
        classify_gci_fail(
            gci_fine=0.001,
            gci_ok=False,
            detail="gci_non_monotonic",
            limit=0.10,
        )
        == "roache_applicability"
    )
    assert (
        classify_gci_fail(
            gci_fine=0.02, gci_ok=True, detail="gci_non_monotonic", limit=0.10
        )
        is None
    )


def test_scorecard_fixture_watch_and_split(tmp_path: Path):
    """Minimal CSV: one watch magnitude fail, one Roache-only noise row."""
    fieldnames = [
        "layout",
        "family",
        "all_mesh_ok",
        "all_trust_ok",
        "governing_fi_gci_fine",
        "governing_fi_gci_ok",
        "governing_fi_gci_detail",
        "peak_abs_n12_gci_fine_skin",
        "peak_abs_n12_gci_ok_skin",
        "peak_abs_n12_gci_detail_skin",
        "peak_abs_n12_gci_fine_web",
        "peak_abs_n12_gci_ok_web",
        "peak_abs_n12_gci_detail_web",
        "peak_mid_sig_gci_fine_skin",
        "peak_mid_sig_gci_ok_skin",
        "peak_mid_sig_gci_detail_skin",
        "peak_mid_sig_gci_fine_web",
        "peak_mid_sig_gci_ok_web",
        "peak_mid_sig_gci_detail_web",
        "peak_hashin_gci_fine_skin",
        "peak_hashin_gci_ok_skin",
        "peak_hashin_gci_detail_skin",
        "peak_hashin_gci_fine_web",
        "peak_hashin_gci_ok_web",
        "peak_hashin_gci_detail_web",
    ]
    rows = [
        {
            "layout": "5D-CN",
            "family": "D",
            "all_mesh_ok": "True",
            "all_trust_ok": "True",
            "governing_fi_gci_fine": "0.12",
            "governing_fi_gci_ok": "False",
            "governing_fi_gci_detail": "asme_roache_gci",
            "peak_abs_n12_gci_fine_skin": "0.179",
            "peak_abs_n12_gci_ok_skin": "False",
            "peak_abs_n12_gci_detail_skin": "",
            "peak_abs_n12_gci_fine_web": "0.02",
            "peak_abs_n12_gci_ok_web": "True",
            "peak_abs_n12_gci_detail_web": "",
            "peak_mid_sig_gci_fine_skin": "0.18",
            "peak_mid_sig_gci_ok_skin": "False",
            "peak_mid_sig_gci_detail_skin": "",
            "peak_mid_sig_gci_fine_web": "0.01",
            "peak_mid_sig_gci_ok_web": "True",
            "peak_mid_sig_gci_detail_web": "",
            "peak_hashin_gci_fine_skin": "0.12",
            "peak_hashin_gci_ok_skin": "False",
            "peak_hashin_gci_detail_skin": "",
            "peak_hashin_gci_fine_web": "0.02",
            "peak_hashin_gci_ok_web": "True",
            "peak_hashin_gci_detail_web": "",
        },
        {
            "layout": "0A",
            "family": "A",
            "all_mesh_ok": "True",
            "all_trust_ok": "True",
            "governing_fi_gci_fine": "0.03",
            "governing_fi_gci_ok": "True",
            "governing_fi_gci_detail": "",
            "peak_abs_n12_gci_fine_skin": "0.0002",
            "peak_abs_n12_gci_ok_skin": "False",
            "peak_abs_n12_gci_detail_skin": "gci_non_monotonic",
            "peak_abs_n12_gci_fine_web": "0.01",
            "peak_abs_n12_gci_ok_web": "True",
            "peak_abs_n12_gci_detail_web": "",
            "peak_mid_sig_gci_fine_skin": "0.001",
            "peak_mid_sig_gci_ok_skin": "True",
            "peak_mid_sig_gci_detail_skin": "",
            "peak_mid_sig_gci_fine_web": "0.01",
            "peak_mid_sig_gci_ok_web": "True",
            "peak_mid_sig_gci_detail_web": "",
            "peak_hashin_gci_fine_skin": "0.03",
            "peak_hashin_gci_ok_skin": "True",
            "peak_hashin_gci_detail_skin": "",
            "peak_hashin_gci_fine_web": "0.02",
            "peak_hashin_gci_ok_web": "True",
            "peak_hashin_gci_detail_web": "",
        },
    ]
    csv_path = tmp_path / "gci_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    card = build_stability_scorecard(csv_path)
    assert card.n_layouts == 2
    assert card.mesh_ok_count == 2 and card.trust_ok_count == 2
    assert card.ci_membrane_pass == 1
    assert card.ci_membrane_magnitude_fail == 1
    assert card.ci_membrane_roache_only == 1
    assert card.watch_key_status["5D-CN"]["ci_magnitude_clear"] is False
    assert any(
        f["classification"] == "magnitude" and f["layout"] == "5D-CN"
        for f in card.magnitude_fails
    )
    assert any(
        f["classification"] == "roache_applicability" and f["layout"] == "0A"
        for f in card.roache_fails
    )
    for key in CI_WATCH_KEYS:
        assert key in card.watch_key_status

    json_path, md_path, _ = write_stability_scorecard(csv_path, out_dir=tmp_path)
    assert json_path.is_file() and md_path.is_file()
    text = md_path.read_text(encoding="utf-8")
    assert "5D-CN" in text and "CI watch keys" in text
