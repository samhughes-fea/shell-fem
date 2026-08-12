"""Full per-element P->S->T->Q shell results, keyed by element ID.

Companion to :mod:`cascade_summary` (which collapses the same cascade into
component-wise peak values for spanwise plotting). This module keeps every
element's raw secondary/tertiary/quaternary output plus its mesh geometry
(global node ids + (y, z) coordinates), for consumers that need the actual
FE mesh data behind a station's plots rather than the compact summary.

Row order across ``secondary``/``tertiary``/``quaternary`` is index-aligned
(see ``ShellTertiaryResultsOrchestrator``/``ShellQuaternaryResultsOrchestrator``
docstrings): row ``i`` in ``secondary.elemental_results.resultant_rows`` is
row ``i`` in every tertiary/quaternary array. Node geometry is looked up
separately via ``(panel_index, local_element_index)`` since that pair -- not
positional index ``i`` -- is the stable key back into ``panel_maps`` and the
formulation cache's ``stiffness_objects`` (secondary silently drops
degenerate elements, e.g. zero-length or missing ABD stack).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

RESULTANT_ATTRS: tuple[str, ...] = ("N11", "N22", "N12", "M11", "M22", "M12", "Q1", "Q2")


def _row_at(arr: Any, i: int) -> Any:
    if arr is None:
        return None
    a = np.asarray(arr)
    if i < 0 or i >= a.shape[0]:
        return None
    return a[i]


def _list_or_none(value: Any) -> Any:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr.tolist()


def _scalar_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v


def _list_or_none_truncated(value: Any, n_ply_true: int | None) -> Any:
    """``_list_or_none``, but sliced to a panel's real ply count first.

    ``ShellTertiaryResultsOrchestrator``/``ShellQuaternaryResultsOrchestrator``
    allocate their per-ply arrays at the *station-wide* max ply count (whichever
    panel is thickest, e.g. the spar cap), then zero-pad every other panel's row
    out to that width -- a 2 mm web reports a 482-slot ply array when only the
    first 16 slots are real plies. Without this truncation, thinner panels would
    silently carry hundreds of fabricated zero-stress/zero-strain "plies".
    """
    lst = _list_or_none(value)
    if lst is None or n_ply_true is None:
        return lst
    return lst[:n_ply_true]


def extract_shell_full_element_results(
    *,
    secondary: Any | None,
    tertiary: Any | None,
    quaternary: Any | None,
    panel_maps: Sequence[Any] | None,
    node_yz: np.ndarray | None,
    stiffness_objects: Sequence[Any] | None = None,
    panel_laminates: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Build one dict per element: geometry + full secondary/tertiary/quaternary state.

    ``panel_laminates`` (``cache.panel_laminates``, indexed by ``panel_index``)
    is used to trim each element's per-ply arrays down to its own panel's real
    ply count -- see :func:`_list_or_none_truncated`. Pass ``None`` to skip
    trimming (per-ply arrays then keep the station-wide padded width).

    Returns an empty list when ``secondary`` (or its elemental results) is
    unavailable -- mirrors :func:`cascade_summary.extract_shell_cascade_summary`'s
    best-effort contract so a missing/partial cascade never raises here.
    """
    rows = []
    if secondary is not None and getattr(secondary, "elemental_results", None) is not None:
        rows = list(getattr(secondary.elemental_results, "resultant_rows", None) or [])
    if not rows:
        return []

    element_id_by_key: dict[tuple[int, int], int] = {}
    for stiff in stiffness_objects or []:
        pi = int(getattr(stiff, "panel_index", -1))
        lei = int(getattr(stiff, "local_element_index", -1))
        element_id_by_key[(pi, lei)] = int(getattr(stiff, "element_id", -1))

    n_ply_true_by_panel: dict[int, int] = {}
    for pi, lam in enumerate(panel_laminates or []):
        plies = getattr(lam, "plies", None)
        if plies is not None:
            n_ply_true_by_panel[pi] = len(plies)

    ply_results = None if tertiary is None else getattr(tertiary, "ply_results", None)
    trusted_mask = None if ply_results is None else getattr(ply_results, "trusted_mask", None)
    shear_trusted_mask = None if ply_results is None else getattr(ply_results, "shear_trusted_mask", None)
    eps0 = None if ply_results is None else getattr(ply_results, "eps0", None)
    kappa = None if ply_results is None else getattr(ply_results, "kappa", None)
    eps_ply_mat = None if ply_results is None else getattr(ply_results, "eps_ply_mat", None)
    sig_ply_mat = None if ply_results is None else getattr(ply_results, "sig_ply_mat", None)
    tau_iface = None if ply_results is None else getattr(ply_results, "tau_iface", None)

    hashin = None if quaternary is None else getattr(quaternary, "hashin_fi", None)
    von_mises_fi = None if quaternary is None else getattr(quaternary, "von_mises_fi", None)
    sig_extreme_fibre = None if quaternary is None else getattr(quaternary, "sig_extreme_fibre", None)
    extreme_fibre_fi = None if quaternary is None else getattr(quaternary, "extreme_fibre_fi", None)
    needs_review = None if quaternary is None else getattr(quaternary, "needs_review", None)

    elements: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        pi_raw = getattr(row, "panel_index", None)
        pi = -1 if pi_raw is None else int(pi_raw)
        local_ei_raw = getattr(row, "station_index", None)
        local_ei = -1 if local_ei_raw is None else int(local_ei_raw)

        node_ids: list[int] = []
        if panel_maps is not None and 0 <= pi < len(panel_maps):
            pm = panel_maps[pi]
            pm_elements = list(getattr(pm, "elements", None) or [])
            gns = list(getattr(pm, "global_nodes", None) or [])
            if 0 <= local_ei < len(pm_elements):
                local_corner_ids = [int(n) for n in pm_elements[local_ei]]
                # pm.elements hold panel-local corner indices; map via global_nodes
                # (same as formulation_factory / MITC4 assembly).
                if gns:
                    node_ids = [
                        int(gns[ln]) if 0 <= ln < len(gns) else -1 for ln in local_corner_ids
                    ]
                else:
                    node_ids = list(local_corner_ids)
        node_coords_yz = None
        if node_yz is not None and node_ids:
            node_coords_yz = [
                _list_or_none(node_yz[nid]) if 0 <= nid < len(node_yz) else None
                for nid in node_ids
            ]

        n_ply_true = n_ply_true_by_panel.get(pi)

        element: dict[str, Any] = {
            "element_id": element_id_by_key.get((pi, local_ei), i),
            "panel_index": pi,
            "panel_label": str(getattr(row, "panel_label", "") or ""),
            "local_element_index": local_ei,
            "node_ids": node_ids,
            "node_coords_yz_m": node_coords_yz,
            "n_ply": n_ply_true,
            "secondary": {
                **{attr: _scalar_or_none(getattr(row, attr, None)) for attr in RESULTANT_ATTRS},
                "thickness_m": _scalar_or_none(getattr(row, "thickness_m", None)),
            },
            "tertiary": None,
            "quaternary": None,
        }

        if ply_results is not None:
            n_iface_true = None if n_ply_true is None else n_ply_true + 1
            element["tertiary"] = {
                "trusted": bool(_row_at(trusted_mask, i)) if trusted_mask is not None else None,
                "shear_trusted": bool(_row_at(shear_trusted_mask, i)) if shear_trusted_mask is not None else None,
                "eps0": _list_or_none(_row_at(eps0, i)),
                "kappa": _list_or_none(_row_at(kappa, i)),
                "eps_ply_mat": _list_or_none_truncated(_row_at(eps_ply_mat, i), n_ply_true),
                "sig_ply_mat": _list_or_none_truncated(_row_at(sig_ply_mat, i), n_ply_true),
                "tau_iface": _list_or_none_truncated(_row_at(tau_iface, i), n_iface_true),
            }

        if quaternary is not None:
            element["quaternary"] = {
                "hashin_in_plane": _list_or_none_truncated(_row_at(None if hashin is None else hashin.in_plane, i), n_ply_true),
                "hashin_interlaminar": _scalar_or_none(_row_at(None if hashin is None else hashin.interlaminar, i)),
                "fi_ft": _list_or_none_truncated(_row_at(None if hashin is None else hashin.fi_ft, i), n_ply_true),
                "fi_fc": _list_or_none_truncated(_row_at(None if hashin is None else hashin.fi_fc, i), n_ply_true),
                "fi_mt": _list_or_none_truncated(_row_at(None if hashin is None else hashin.fi_mt, i), n_ply_true),
                "fi_mc": _list_or_none_truncated(_row_at(None if hashin is None else hashin.fi_mc, i), n_ply_true),
                "von_mises_fi": _scalar_or_none(_row_at(von_mises_fi, i)),
                "sig_extreme_fibre": _list_or_none(_row_at(sig_extreme_fibre, i)),
                "extreme_fibre_fi": _scalar_or_none(_row_at(extreme_fibre_fi, i)),
                "needs_review": bool(_row_at(needs_review, i)) if needs_review is not None else None,
            }

        elements.append(element)

    return elements


__all__ = ["extract_shell_full_element_results"]
