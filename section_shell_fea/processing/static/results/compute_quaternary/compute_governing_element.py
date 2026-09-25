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


def _bundle_zones(bundle: Any) -> list:
    """The bundle's material zones, or raise saying which attribute is missing.

    Deliberately not ``getattr(bundle, "zones", None) or []``. Both functions
    below used to read ``bundle.panels``, which the ``panels`` -> ``zones``
    rename left behind: the attribute was simply absent, ``or []`` supplied an
    empty list, the loop body never ran, and a governing FI of ``0.0`` was
    returned for every bundle. Nothing raised, and reported job FI was zero
    everywhere (docs/bug_log.md BL-012).

    An empty zone list is still legal — a bundle may genuinely carry none — so
    the distinction that matters is *absent attribute* versus *empty sequence*,
    which is exactly what the swallow destroyed.
    """
    zones = getattr(bundle, "zones", None)
    if zones is None:
        raise AttributeError(
            f"{type(bundle).__name__} has no 'zones'. Governing-FI selection "
            f"needs the material zones; a silent empty list here returns FI=0 "
            f"for the whole section (BL-012)."
        )
    return list(zones)


def max_hashin_fi_from_bundle(
    bundle: Any,
    *,
    strengths: dict[str, float] | None = None,
) -> tuple[float, bool]:
    """Governing in-plane Hashin FI over all trusted panel elements.

    Strengths come from each panel's laminate (same path as
    :func:`select_governing_hashin_element`). Passing ``strengths=`` overrides
    with an explicit dict (tests / demos only). Silent substitution of
    ``default_skin_strengths_pa`` is refused (FEA-FI-001).
    """
    from blade_structure_utilities.material_utilities.laminate import (
        LaminateDefinition,
        laminate_definition_from_legacy,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.api.fi_recovery import (
        hashin_fi_from_row,
        hashin_fi_per_ply_from_row,
    )

    zones = _bundle_zones(bundle)
    all_res = getattr(bundle, "all_panel_results", None) or []
    fi_max = 0.0
    needs_review = False
    for pi, panel in enumerate(zones):
        if pi >= len(all_res):
            continue
        plies = panel.lam.build_plies()
        lam_raw = laminate_definition_from_legacy(panel.lam, plies=plies)
        for row in all_res[pi] or []:
            if row is None:
                continue
            if strengths is not None:
                fi, nr = hashin_fi_from_row(row, plies, lam=lam_raw, **strengths)
                if nr:
                    needs_review = True
                    continue
                fi_max = max(fi_max, float(fi))
                continue
            ld = getattr(getattr(panel, "lam", None), "_ld", getattr(panel, "lam", None))
            if not isinstance(ld, LaminateDefinition) or not ld.plies:
                if isinstance(lam_raw, LaminateDefinition) and lam_raw.plies:
                    ld = lam_raw
                else:
                    raise ValueError(
                        f"panel {pi} has no LaminateDefinition with plies; cannot "
                        "evaluate Hashin without producer strengths (FEA-FI-001). "
                        "Pass strengths= explicitly only for demos."
                    )
            fi_v, nr = hashin_fi_per_ply_from_row(row, ld, plies)
            if nr:
                needs_review = True
                continue
            if fi_v.size == 0:
                continue
            fi_max = max(fi_max, float(np.max(fi_v)))
    return fi_max, needs_review


def select_governing_hashin_element(
    bundle: Any,
    *,
    composite_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Governing element over panels x elements x plies with provenance audit."""
    from blade_structure_utilities.material_utilities.laminate import LaminateDefinition
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.api.fi_recovery import hashin_fi_per_ply_from_row

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
    zones = _bundle_zones(bundle)
    all_res = getattr(bundle, "all_panel_results", None) or []
    for pi, panel in enumerate(zones):
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
