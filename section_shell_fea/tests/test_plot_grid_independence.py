"""Offline smoke for GCI study plotter."""

from __future__ import annotations

import csv
from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.plot_grid_independence import (
    plot_grid_independence_study,
)


def test_plot_grid_independence_from_tiny_csv(tmp_path: Path):
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
                "peak_hashin_gci_fine_skin": "",
                "peak_hashin_gci_fine_web": "",
            }
        )
    paths = plot_grid_independence_study(tmp_path, backend_label="TEST", write_hashin=False)
    assert paths
    assert (tmp_path / "plots" / "gci_fine_peak_abs_n12_by_layout.png").is_file()
    assert (tmp_path / "plots" / "ladder_peak_abs_n12_skin.png").is_file()
