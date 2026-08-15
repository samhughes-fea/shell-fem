"""Offline smoke for GCI study plotter."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.plot_grid_independence import (
    plot_grid_independence_study,
    plot_study_layout_convergence,
    shell_from_recipe,
)


def _write_tiny_study(tmp_path: Path, *, with_recipe: bool = False) -> None:
    ladder = tmp_path / "gci_ladder.csv"
    summary = tmp_path / "gci_summary.csv"
    with ladder.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "layout",
                "frac",
                "h_m",
                "peak_abs_n12_skin",
                "peak_abs_n12_cap",
                "peak_abs_n12_web",
                "peak_mid_sig_skin",
                "peak_mid_sig_web",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "layout": "2A-F",
                "frac": "0.024",
                "h_m": "0.006",
                "peak_abs_n12_skin": "100",
                "peak_abs_n12_cap": "0",
                "peak_abs_n12_web": "50",
                "peak_mid_sig_skin": "1e7",
                "peak_mid_sig_web": "5e6",
            }
        )
        w.writerow(
            {
                "layout": "2A-F",
                "frac": "0.012",
                "h_m": "0.003",
                "peak_abs_n12_skin": "110",
                "peak_abs_n12_cap": "0",
                "peak_abs_n12_web": "55",
                "peak_mid_sig_skin": "1.1e7",
                "peak_mid_sig_web": "5.5e6",
            }
        )
    with summary.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "layout",
                "peak_abs_n12_gci_fine_skin",
                "peak_abs_n12_gci_fine_web",
                "peak_abs_n12_gci_detail_skin",
                "peak_abs_n12_gci_detail_web",
                "peak_abs_n12_last_two_skin",
                "peak_abs_n12_last_two_web",
                "peak_mid_sig_gci_fine_skin",
                "peak_mid_sig_gci_fine_web",
                "peak_hashin_gci_fine_skin",
                "peak_hashin_gci_fine_web",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "layout": "2A-F",
                "peak_abs_n12_gci_fine_skin": "0.05",
                "peak_abs_n12_gci_fine_web": "0.04",
                "peak_abs_n12_gci_detail_skin": "ok",
                "peak_abs_n12_gci_detail_web": "ok",
                "peak_abs_n12_last_two_skin": "1.1",
                "peak_abs_n12_last_two_web": "1.05",
                "peak_mid_sig_gci_fine_skin": "0.06",
                "peak_mid_sig_gci_fine_web": "0.05",
                "peak_hashin_gci_fine_skin": "",
                "peak_hashin_gci_fine_web": "",
            }
        )
    if with_recipe:
        (tmp_path / "station_recipe.json").write_text(
            json.dumps(
                {
                    "chord_m": 0.24,
                    "twist_deg": 34.0,
                    "thickness": {"skin": 0.001, "spar": 0.002, "web": 0.0015},
                }
            )
            + "\n",
            encoding="utf-8",
        )


def test_plot_grid_independence_from_tiny_csv(tmp_path: Path):
    _write_tiny_study(tmp_path, with_recipe=True)
    paths = plot_grid_independence_study(tmp_path, backend_label="TEST", write_hashin=False)
    assert paths
    assert (tmp_path / "plots" / "gci_fine_peak_abs_n12_by_layout.png").is_file()
    assert (tmp_path / "plots" / "ladder_peak_abs_n12_skin.png").is_file()
    assert (tmp_path / "plots" / "layouts" / "A" / "2A-F_convergence.png").is_file()


def test_layout_convergence_with_geometry_smoke(tmp_path: Path):
    pytest.importorskip("matplotlib")
    _write_tiny_study(tmp_path, with_recipe=True)
    shell = shell_from_recipe("2A-F", tmp_path / "station_recipe.json")
    assert shell is not None
    assert list(getattr(shell, "midlines", []) or [])
    paths = plot_study_layout_convergence(tmp_path, keys=("2A-F",))
    assert paths
    assert paths[0].is_file()
    assert paths[0].parent.name == "A"
    assert paths[0].stat().st_size > 1000
