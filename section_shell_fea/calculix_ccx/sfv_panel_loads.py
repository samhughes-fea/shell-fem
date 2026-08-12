"""Map ``SectionForceVector`` contour densities onto CCX ``nx`` / ``nxy`` panel arrays.

MITC4 applies SFV samples as nodal loads via
``section_force_vector_to_nodal_load``. The CCX deck writer expects per-panel
section-force densities along ``panel.s`` (N/m), which ``_cload_entries`` then
lumps onto strip nodes. This adapter resamples ``f_x`` → ``N11`` and ``f_s`` →
``N12`` (thin-wall ``q``) onto each panel contour.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.load_utilities.section_loads import (
    ContourForceSample,
    SectionForceVector,
)


def _panel_yz_at_s(panel: Any, s_query: NDArray[np.float64]) -> NDArray[np.float64]:
    """Interpolate panel midline ``(y, z)`` at contour stations ``s_query``."""
    s_p = np.asarray(getattr(panel, "s"), dtype=float).ravel()
    nodes = np.asarray(getattr(panel, "nodes"), dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] < 2 or s_p.size < 1:
        return np.zeros((s_query.size, 2), dtype=float)
    if s_p.size == 1:
        return np.repeat(nodes[0:1, :2], s_query.size, axis=0)
    y = np.interp(s_query, s_p, nodes[:, 0])
    z = np.interp(s_query, s_p, nodes[:, 1])
    return np.column_stack([y, z])


def _kind_of(label: str) -> str:
    return label.split(":", 1)[0] if ":" in label else "panel"


def _samples_for_panel(
    samples: Sequence[ContourForceSample],
    panel: Any,
) -> list[ContourForceSample]:
    """Prefer exact label match; else same kind; else all samples."""
    lab = str(getattr(panel, "label", ""))
    by_label = [s for s in samples if str(s.label) == lab]
    if by_label:
        return by_label
    # Hierarchical labels: SFV may use a parent strip name contained in panel label.
    contained = [s for s in samples if str(s.label) and str(s.label) in lab]
    if contained:
        return contained
    kind = _kind_of(lab)
    by_kind = [s for s in samples if str(s.kind) == kind]
    return by_kind if by_kind else list(samples)


def sfv_to_nx_nxy(
    sfv: SectionForceVector,
    panels: Sequence[Any],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Resample SFV ``f_x`` / ``f_s`` onto each panel's ``s`` stations.

    Returns
    -------
    nx_panels, nxy_panels
        Lists of ``float64`` arrays aligned with ``panel.s`` (N/m).
    """
    samples = tuple(sfv.samples)
    nx_panels: list[NDArray[np.float64]] = []
    nxy_panels: list[NDArray[np.float64]] = []

    for panel in panels:
        s = np.asarray(getattr(panel, "s"), dtype=float).ravel()
        if s.size == 0:
            nx_panels.append(np.zeros(0, dtype=float))
            nxy_panels.append(np.zeros(0, dtype=float))
            continue
        if not samples:
            nx_panels.append(np.zeros_like(s))
            nxy_panels.append(np.zeros_like(s))
            continue

        cand = _samples_for_panel(samples, panel)
        sy = np.asarray([float(sm.y) for sm in cand], dtype=float)
        sz = np.asarray([float(sm.z) for sm in cand], dtype=float)
        fx = np.asarray([float(sm.f_x) for sm in cand], dtype=float)
        fs = np.asarray([float(sm.f_s) for sm in cand], dtype=float)
        yz = _panel_yz_at_s(panel, s)
        nx = np.zeros(s.size, dtype=float)
        nxy = np.zeros(s.size, dtype=float)
        for i in range(s.size):
            d2 = (sy - yz[i, 0]) ** 2 + (sz - yz[i, 1]) ** 2
            j = int(np.argmin(d2))
            nx[i] = fx[j]
            nxy[i] = fs[j]
        nx_panels.append(nx)
        nxy_panels.append(nxy)
    return nx_panels, nxy_panels


__all__ = ["sfv_to_nx_nxy"]
