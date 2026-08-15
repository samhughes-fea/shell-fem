"""CLI: locate cap peaks and compare raw max vs robust/interior stats.

Default load source is Cotide ``new_3.2m_rotor`` + ``NACA_63-415.dat``.

Examples::

    python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_cap_peak_sampling
    python -m ...run_cap_peak_sampling --keys 2B-CN,2A-F
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cap_peak_sampling import (
    DEFAULT_INVESTIGATION_KEYS,
    run_cap_peak_study,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cap-peak sampling investigation (raw max vs p95 / interior)."
    )
    parser.add_argument(
        "--keys",
        default=",".join(DEFAULT_INVESTIGATION_KEYS),
        help="Comma-separated layout keys (default: investigation set)",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "blade_structure_utilities/fem_utilities/section_shell_fea/"
            "visualisation/output/cap_peak_sampling"
        ),
        help="Directory for ladder / locations / drift CSVs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-layout / per-rung progress",
    )
    parser.add_argument(
        "--interface",
        choices=(
            "shared",
            "transformed_basis",
            "shared_rotated",
            "hybrid_collinear",
            "hybrid_kind_crossing",
            "shared_cluster_frame",
        ),
        default="shared",
        help="Junction constraint mode (default: shared)",
    )
    parser.add_argument(
        "--fracs",
        default="",
        help="Optional comma-separated density fracs (default: study ladder)",
    )
    args = parser.parse_args(argv)
    keys = tuple(k.strip() for k in str(args.keys).split(",") if k.strip())
    fracs = tuple(float(x) for x in str(args.fracs).split(",") if x.strip())
    study_kw: dict = {
        "out_dir": Path(args.out_dir),
        "progress": not args.quiet,
        "interface_constraint_mode": args.interface,
    }
    if fracs:
        study_kw["refinements"] = fracs
    _by, drifts = run_cap_peak_study(keys, **study_kw)
    print(f"wrote {Path(args.out_dir) / 'cap_peak_ladder.csv'}", flush=True)
    print(f"wrote {Path(args.out_dir) / 'cap_peak_locations.json'}", flush=True)
    print(f"wrote {Path(args.out_dir) / 'cap_peak_drift_summary.csv'}", flush=True)

    print("Verdict summary (cap peak_hashin max):", flush=True)
    any_artifact = False
    any_bulk = False
    for d in drifts:
        if not (d.kind == "cap" and d.metric == "peak_hashin" and d.statistic == "max"):
            continue
        prod = f"{d.production_drift:.3f}" if d.production_drift == d.production_drift else "nan"
        lt = f"{d.last_two_drift:.3f}" if d.last_two_drift == d.last_two_drift else "nan"
        print(
            f"  {d.layout:8s}  prod={prod}  last_two={lt}  {d.notes}",
            flush=True,
        )
        if str(d.notes).startswith("sampling_artifact"):
            any_artifact = True
        if d.notes == "bulk_nonconvergence":
            any_bulk = True

    # Also show robust counterparts for failing layouts
    print("Robust counterparts (cap peak_hashin p95 / interior_max):", flush=True)
    for d in drifts:
        if d.kind != "cap" or d.metric != "peak_hashin":
            continue
        if d.statistic not in ("p95", "interior_max"):
            continue
        prod = f"{d.production_drift:.3f}" if d.production_drift == d.production_drift else "nan"
        lt = f"{d.last_two_drift:.3f}" if d.last_two_drift == d.last_two_drift else "nan"
        print(
            f"  {d.layout:8s}  {d.statistic:14s}  prod={prod}  last_two={lt}",
            flush=True,
        )

    if any_bulk and not any_artifact:
        print(
            "OVERALL: bulk non-convergence "
            "(raw max and last-two p95/interior both drift >2).",
            flush=True,
        )
        return 1
    if any_artifact:
        print(
            "OVERALL: sampling artifact - raw max drifts at interface-tagged elements "
            "while last-two p95/interior stay within band; "
            "next step is acceptance-metric change (exclude endpoint band / use p95).",
            flush=True,
        )
        return 0
    print("OVERALL: no cap Hashin max gate-class failures in this set.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
