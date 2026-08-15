"""Path-compare helpers moved out of ``api/fi_recovery`` (R15)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellPanelResultants


@dataclass(frozen=True)
class PathCompareStats:
    """Per-field relative L2 error between two solve paths on matched elements."""

    rel_l2: dict[str, float]
    max_hashin_b: float
    max_hashin_c: float
    hashin_needs_review_b: bool
    hashin_needs_review_c: bool


def _matched_element_rows(
    bundle_b: Any,
    bundle_c: Any,
    panel_index: int = 0,
) -> tuple[list[ShellPanelResultants], list[ShellPanelResultants]]:
    rb = getattr(bundle_b, "all_panel_results", None) or [[]]
    rc = getattr(bundle_c, "all_panel_results", None) or [[]]
    if panel_index >= len(rb) or panel_index >= len(rc):
        return [], []
    pb = [r for r in (rb[panel_index] or []) if r is not None]
    pc = [r for r in (rc[panel_index] or []) if r is not None]
    n = min(len(pb), len(pc))
    return pb[:n], pc[:n]


def compare_path_resultants(
    bundle_b: Any,
    bundle_c: Any,
    *,
    fields: Iterable[str] = ("N11", "N22", "N12", "M11", "M22", "M12"),
    panel_index: int = 0,
    strengths: dict[str, float] | None = None,
) -> PathCompareStats:
    """Compare two recovery bundles on one panel (relative L2 per field)."""
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.api.fi_recovery import max_hashin_fi_from_bundle

    rows_b, rows_c = _matched_element_rows(bundle_b, bundle_c, panel_index)
    rel: dict[str, float] = {}
    for fld in fields:
        vb = np.array([float(getattr(r, fld)) for r in rows_b], dtype=float)
        vc = np.array([float(getattr(r, fld)) for r in rows_c], dtype=float)
        if vb.size == 0:
            rel[fld] = float("nan")
            continue
        num = float(np.linalg.norm(vb - vc))
        den = max(float(np.linalg.norm(vb)), 1e-12)
        rel[fld] = num / den
    hb, nrb = max_hashin_fi_from_bundle(bundle_b, strengths=strengths)
    hc, nrc = max_hashin_fi_from_bundle(bundle_c, strengths=strengths)
    return PathCompareStats(
        rel_l2=rel,
        max_hashin_b=hb,
        max_hashin_c=hc,
        hashin_needs_review_b=nrb,
        hashin_needs_review_c=nrc,
    )


def section_integrals_from_bundle(
    bundle: Any,
    *,
    fields: Iterable[str] = ("N11", "N22", "N12"),
) -> dict[str, float]:
    """Contour integrals ``∮ N ds`` per field over all panels (trapezoid on ``s``)."""
    # ``.zones``, not ``.panels``: the rename left this reading an attribute the
    # bundle no longer has, and ``or []`` turned that into "no panels" — every
    # contour integral came back 0.0 with nothing said (docs/bug_log.md BL-012).
    zones = getattr(bundle, "zones", None)
    if zones is None:
        raise AttributeError(
            f"{type(bundle).__name__} has no 'zones'; contour integrals over an "
            f"empty sequence are silently zero (BL-012)."
        )
    all_res = getattr(bundle, "all_panel_results", None) or []
    out = {f: 0.0 for f in fields}
    for pi, panel in enumerate(zones):
        if pi >= len(all_res):
            continue
        s = np.asarray(panel.s, dtype=float).ravel()
        rows = all_res[pi] or []
        if len(s) < 2 or len(rows) < 2:
            continue
        n_elem = min(len(rows), len(s) - 1)
        for ei in range(n_elem):
            row = rows[ei]
            if row is None:
                continue
            ds = float(s[ei + 1] - s[ei])
            for f in fields:
                out[f] += float(getattr(row, f)) * ds
    return out


__all__ = [
    "PathCompareStats",
    "compare_path_resultants",
    "section_integrals_from_bundle",
]
