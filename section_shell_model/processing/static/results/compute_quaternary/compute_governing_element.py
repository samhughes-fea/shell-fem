"""Tier 4: governing panel x element x ply selection, provenance audit.

Moved from ``api/fi_recovery.select_governing_hashin_element`` /
``max_hashin_fi_from_bundle``. Both still evaluate Hashin per-row via the
single-row facade in ``api/fi_recovery.py`` (lazily imported to avoid an
api <-> compute_quaternary import cycle, matching the existing lazy-import
convention used elsewhere in this package).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def max_hashin_fi_from_bundle(
    bundle: Any,
    *,
    strengths: dict[str, float] | None = None,
) -> tuple[float, bool]:
    """Governing in-plane Hashin FI over all trusted panel elements."""
    from blade_structure_utilities.section_materials.laminate import laminate_definition_from_legacy
    from blade_structure_utilities.section_shell_model.processing.static.results.api.fi_recovery import (
        default_skin_strengths_pa,
        hashin_fi_from_row,
    )

    st = strengths or default_skin_strengths_pa()
    panels = getattr(bundle, "panels", None) or []
    all_res = getattr(bundle, "all_panel_results", None) or []
    fi_max = 0.0
    needs_review = False
    for pi, panel in enumerate(panels):
        if pi >= len(all_res):
            continue
        plies = panel.lam.build_plies()
        lam_raw = laminate_definition_from_legacy(panel.lam, plies=plies)
        for row in all_res[pi] or []:
            if row is None:
                continue
            fi, nr = hashin_fi_from_row(row, plies, lam=lam_raw, **st)
            if nr:
                needs_review = True
                continue
            fi_max = max(fi_max, fi)
    return fi_max, needs_review


def select_governing_hashin_element(
    bundle: Any,
    *,
    composite_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Governing element over panels x elements x plies with provenance audit."""
    from blade_structure_utilities.section_materials.laminate import LaminateDefinition
    from blade_structure_utilities.section_shell_model.processing.static.results.api.fi_recovery import hashin_fi_per_ply_from_row

    best_fi = 0.0
    gov: dict[str, Any] = {
        "fi": 0.0,
        "panel_index": -1,
        "element_index": -1,
        "ply_index": -1,
        "subcomponent": "",
        "needs_review": False,
        "provenance_kinds": [],
    }
    panels = getattr(bundle, "panels", None) or []
    all_res = getattr(bundle, "all_panel_results", None) or []
    for pi, panel in enumerate(panels):
        pnm = str(getattr(panel, "name", f"p{pi}"))
        if pi >= len(all_res):
            continue
        lam_raw = getattr(panel, "lam", None)
        ld = getattr(lam_raw, "_ld", lam_raw)
        if not isinstance(ld, LaminateDefinition) or not ld.plies:
            continue
        plies = panel.lam.build_plies()
        for ei, row in enumerate(all_res[pi] or []):
            if row is None:
                continue
            fi_v, nr = hashin_fi_per_ply_from_row(row, ld, plies)
            if nr:
                gov["needs_review"] = True
                continue
            if fi_v.size == 0:
                continue
            k = int(np.argmax(fi_v))
            fv = float(fi_v[k])
            if fv > best_fi:
                best_fi = fv
                kinds = sorted({p.kind.value for p in row.provenance.values()})
                gov.update(
                    fi=fv,
                    panel_index=pi,
                    element_index=ei,
                    ply_index=k,
                    subcomponent=pnm,
                    provenance_kinds=kinds,
                )
    return gov


__all__ = ["max_hashin_fi_from_bundle", "select_governing_hashin_element"]
