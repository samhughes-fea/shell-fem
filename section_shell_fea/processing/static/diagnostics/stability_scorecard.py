"""Stability scorecard from ``gci_summary.csv`` (machine + human).

Classifies CI/study GCI fails as:

- ``magnitude`` — fine |GCI| exceeds the gate limit
- ``roache_applicability`` — ``gci_non_monotonic`` (or similar) with |GCI| ≤ limit

Day-to-day summaries emphasise magnitude fails; Roache rows stay visible but
do not dominate the residual set.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.grid_independence_study import (
    CI_KIND_METRICS,
    CI_MAX_GCI_FINE,
    CI_WATCH_KEYS,
    INFORMATIONAL_KINDS,
    STUDY_FI_WATCH_KEYS,
    STUDY_MAX_GCI_FINE,
)

_CI_KINDS = ("skin", "web")  # cap is informational under CI
_STUDY_KINDS = ("skin", "web")


@dataclass(frozen=True)
class GciFailRow:
    """One gated GCI cell that did not pass applicability + magnitude."""

    layout: str
    family: str
    metric: str
    kind: str
    gci_fine: float
    limit: float
    detail: str
    classification: str  # magnitude | roache_applicability | undefined


@dataclass
class StabilityScorecard:
    """Aggregated stability package surface for a study directory."""

    study_dir: str
    n_layouts: int
    mesh_ok_count: int
    trust_ok_count: int
    ci_membrane_limit: float = CI_MAX_GCI_FINE
    study_fi_limit: float = STUDY_MAX_GCI_FINE
    ci_membrane_pass: int = 0
    ci_membrane_magnitude_fail: int = 0
    ci_membrane_roache_only: int = 0
    study_hashin_pass: int = 0
    study_gov_pass: int = 0
    family_tallies: dict[str, dict[str, int]] = field(default_factory=dict)
    watch_key_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    study_fi_watch_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    magnitude_fails: list[dict[str, Any]] = field(default_factory=list)
    roache_fails: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(val: str | None) -> float:
    if val is None or val == "" or val.lower() in ("nan", "none"):
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def _bool(val: str | None) -> bool:
    return str(val).strip().lower() in ("true", "1", "yes")


def classify_gci_fail(
    *,
    gci_fine: float,
    gci_ok: bool,
    detail: str,
    limit: float,
) -> str | None:
    """Return classification if the cell is a fail; else None.

    Magnitude dominates: |GCI| > limit is always ``magnitude`` even when
    ``gci_ok`` is True (defensive). Small-|GCI| ``gci_non_monotonic`` is
    ``roache_applicability``.
    """
    mag = abs(gci_fine) if gci_fine == gci_fine else float("nan")
    if mag == mag and mag > limit:
        return "magnitude"
    if gci_ok:
        return None
    detail_l = (detail or "").strip().lower()
    if "undefined" in detail_l:
        return "undefined"
    if "non_monotonic" in detail_l or mag == mag:
        return "roache_applicability"
    return "undefined"


def _metric_cells(
    row: dict[str, str],
    metrics: Sequence[str],
    kinds: Sequence[str],
    *,
    limit: float,
    skip_kinds: Iterable[str] = (),
) -> list[GciFailRow]:
    skip = set(skip_kinds) | set(INFORMATIONAL_KINDS)
    layout = row["layout"]
    family = row.get("family", "")
    out: list[GciFailRow] = []
    for metric in metrics:
        for kind in kinds:
            if kind in skip:
                continue
            ok_key = f"{metric}_gci_ok_{kind}"
            if ok_key not in row:
                continue
            gci = _f(row.get(f"{metric}_gci_fine_{kind}"))
            ok = _bool(row.get(ok_key))
            detail = str(row.get(f"{metric}_gci_detail_{kind}", "") or "")
            cls = classify_gci_fail(
                gci_fine=gci, gci_ok=ok, detail=detail, limit=limit
            )
            if cls is None:
                continue
            out.append(
                GciFailRow(
                    layout=layout,
                    family=family,
                    metric=metric,
                    kind=kind,
                    gci_fine=gci,
                    limit=limit,
                    detail=detail,
                    classification=cls,
                )
            )
    return out


def build_stability_scorecard(
    summary_csv: Path | str,
    *,
    study_dir: Path | str | None = None,
) -> StabilityScorecard:
    """Build a scorecard from ``gci_summary.csv``."""
    path = Path(summary_csv)
    root = Path(study_dir) if study_dir is not None else path.parent
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    card = StabilityScorecard(
        study_dir=str(root),
        n_layouts=len(rows),
        mesh_ok_count=sum(1 for r in rows if _bool(r.get("all_mesh_ok"))),
        trust_ok_count=sum(1 for r in rows if _bool(r.get("all_trust_ok"))),
        notes=[
            "CI membrane = peak_abs_n12 + peak_mid_sig on skin/web; cap informational.",
            "Magnitude fails dominate day-to-day; Roache applicability stays in CSV.",
            (
                f"Full --keys all --gates ci is expected to fail CI_WATCH_KEYS="
                f"{list(CI_WATCH_KEYS)} until Phase B clears them."
            ),
        ],
    )

    families = sorted({str(r.get("family", "?")) for r in rows})
    for fam in families:
        card.family_tallies[fam] = {
            "n": 0,
            "mesh_ok": 0,
            "trust_ok": 0,
            "ci_membrane_pass": 0,
            "ci_magnitude_fail": 0,
            "study_hashin_pass": 0,
            "study_gov_pass": 0,
        }

    mag_fails: list[GciFailRow] = []
    roache_fails: list[GciFailRow] = []

    for row in rows:
        fam = str(row.get("family", "?"))
        layout = row["layout"]
        ft = card.family_tallies.setdefault(
            fam,
            {
                "n": 0,
                "mesh_ok": 0,
                "trust_ok": 0,
                "ci_membrane_pass": 0,
                "ci_magnitude_fail": 0,
                "study_hashin_pass": 0,
                "study_gov_pass": 0,
            },
        )
        ft["n"] += 1
        if _bool(row.get("all_mesh_ok")):
            ft["mesh_ok"] += 1
        if _bool(row.get("all_trust_ok")):
            ft["trust_ok"] += 1

        ci_cells = _metric_cells(
            row, CI_KIND_METRICS, _CI_KINDS, limit=CI_MAX_GCI_FINE
        )
        ci_mag = [c for c in ci_cells if c.classification == "magnitude"]
        ci_roache = [c for c in ci_cells if c.classification == "roache_applicability"]
        mag_fails.extend(ci_mag)
        roache_fails.extend(ci_roache)

        if not ci_mag:
            card.ci_membrane_pass += 1
            ft["ci_membrane_pass"] += 1
        else:
            card.ci_membrane_magnitude_fail += 1
            ft["ci_magnitude_fail"] += 1
        if ci_roache and not ci_mag:
            card.ci_membrane_roache_only += 1

        hashin_cells = _metric_cells(
            row, ("peak_hashin",), _STUDY_KINDS, limit=STUDY_MAX_GCI_FINE
        )
        hashin_mag = [c for c in hashin_cells if c.classification == "magnitude"]
        if not hashin_mag:
            card.study_hashin_pass += 1
            ft["study_hashin_pass"] += 1

        gov_gci = _f(row.get("governing_fi_gci_fine"))
        gov_ok = _bool(row.get("governing_fi_gci_ok"))
        gov_detail = str(row.get("governing_fi_gci_detail", "") or "")
        gov_cls = classify_gci_fail(
            gci_fine=gov_gci,
            gci_ok=gov_ok,
            detail=gov_detail,
            limit=STUDY_MAX_GCI_FINE,
        )
        if gov_cls != "magnitude":
            card.study_gov_pass += 1
            ft["study_gov_pass"] += 1

        if layout in CI_WATCH_KEYS:
            card.watch_key_status[layout] = {
                "ci_magnitude_clear": not ci_mag,
                "ci_magnitude_fails": [asdict(c) for c in ci_mag],
                "ci_roache_fails": [asdict(c) for c in ci_roache],
                "mesh_ok": _bool(row.get("all_mesh_ok")),
                "trust_ok": _bool(row.get("all_trust_ok")),
            }
        if layout in STUDY_FI_WATCH_KEYS:
            card.study_fi_watch_status[layout] = {
                "hashin_magnitude_clear": not hashin_mag,
                "gov_magnitude_clear": gov_cls != "magnitude",
                "governing_fi_gci_fine": gov_gci,
            }

    # Ensure watch keys appear even if missing from CSV.
    for key in CI_WATCH_KEYS:
        card.watch_key_status.setdefault(
            key,
            {
                "ci_magnitude_clear": False,
                "ci_magnitude_fails": [],
                "ci_roache_fails": [],
                "mesh_ok": False,
                "trust_ok": False,
                "missing_from_summary": True,
            },
        )

    card.magnitude_fails = [asdict(c) for c in mag_fails]
    card.roache_fails = [asdict(c) for c in roache_fails]
    return card


def render_scorecard_markdown(card: StabilityScorecard) -> str:
    """Short human table for docs / PRs."""
    lines = [
        "# Stability scorecard",
        "",
        f"Study dir: `{card.study_dir}`",
        "",
        "## Headline",
        "",
        f"| Check | Pass |",
        f"|-------|------|",
        f"| Mesh OK | {card.mesh_ok_count}/{card.n_layouts} |",
        f"| Trust OK | {card.trust_ok_count}/{card.n_layouts} |",
        f"| CI membrane magnitude (≤{card.ci_membrane_limit:.0%}) | "
        f"{card.ci_membrane_pass}/{card.n_layouts} |",
        f"| CI membrane Roache-only layouts | {card.ci_membrane_roache_only} |",
        f"| Study Hashin magnitude (≤{card.study_fi_limit:.0%}) | "
        f"{card.study_hashin_pass}/{card.n_layouts} |",
        f"| Study governing_fi magnitude (≤{card.study_fi_limit:.0%}) | "
        f"{card.study_gov_pass}/{card.n_layouts} |",
        "",
        "## CI watch keys",
        "",
        "| Layout | Magnitude clear | Notes |",
        "|--------|-----------------|-------|",
    ]
    for key in CI_WATCH_KEYS:
        st = card.watch_key_status.get(key, {})
        clear = bool(st.get("ci_magnitude_clear"))
        fails = st.get("ci_magnitude_fails") or []
        note = "OK" if clear else ", ".join(
            f"{f.get('metric')}/{f.get('kind')}={float(f.get('gci_fine', float('nan'))):.1%}"
            for f in fails
        )
        if st.get("missing_from_summary"):
            note = "missing from summary"
        lines.append(f"| `{key}` | {'yes' if clear else 'no'} | {note} |")

    lines.extend(
        [
            "",
            "## Per family",
            "",
            "| Family | n | Mesh | Trust | CI mag pass | Hashin | Gov |",
            "|--------|---|------|-------|-------------|--------|-----|",
        ]
    )
    for fam, ft in sorted(card.family_tallies.items()):
        lines.append(
            f"| {fam} | {ft['n']} | {ft['mesh_ok']}/{ft['n']} | "
            f"{ft['trust_ok']}/{ft['n']} | {ft['ci_membrane_pass']}/{ft['n']} | "
            f"{ft['study_hashin_pass']}/{ft['n']} | {ft['study_gov_pass']}/{ft['n']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for n in card.notes:
        lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines)


def write_stability_scorecard(
    summary_csv: Path | str,
    *,
    out_dir: Path | str | None = None,
) -> tuple[Path, Path, StabilityScorecard]:
    """Write ``stability_scorecard.json`` + ``.md`` next to the study summary."""
    path = Path(summary_csv)
    root = Path(out_dir) if out_dir is not None else path.parent
    root.mkdir(parents=True, exist_ok=True)
    card = build_stability_scorecard(path, study_dir=root)
    json_path = root / "stability_scorecard.json"
    md_path = root / "stability_scorecard.md"
    json_path.write_text(json.dumps(card.to_dict(), indent=2, allow_nan=True), encoding="utf-8")
    md_path.write_text(render_scorecard_markdown(card), encoding="utf-8")
    return json_path, md_path, card


__all__ = [
    "GciFailRow",
    "StabilityScorecard",
    "build_stability_scorecard",
    "classify_gci_fail",
    "render_scorecard_markdown",
    "write_stability_scorecard",
]
