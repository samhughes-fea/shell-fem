"""Map Channel-C :class:`SectionForceVector` contour samples onto CCX panel N11/N12.

Identity rule (same as FEA-L-001 / :func:`_map_loads_to_panel`): membership by
strip **label** only. Samples from another strip must never bleed onto a panel
via kind-wide or global nearest-YZ fallbacks (FEA-CCX-001).

Within a matched strip, ``f_x`` / ``f_s`` are interpolated in arc-length along
the panel polyline onto each mesh station ``s``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.load_utilities.section_loads import ContourForceSample, SectionForceVector

# Unified with FEA-L-001 / FEA-CT-001 coincidence contracts.
_COINCIDENCE_TOL_M = 1e-9


def _strip_label(panel_label: str) -> str:
    lab = str(panel_label)
    if lab.startswith(("skin:", "web:", "cap:")):
        return lab.partition(":")[2]
    return lab


def _samples_for_panel_label(
    samples: Sequence[ContourForceSample],
    panel_label: str,
) -> list[ContourForceSample]:
    lab = str(panel_label)
    strip = _strip_label(lab)
    return [s for s in samples if str(s.label) in (lab, strip)]


def _panel_arc_prefix(nodes: NDArray[np.floating]) -> NDArray[np.float64]:
    n = np.asarray(nodes, dtype=np.float64)
    if n.ndim != 2 or n.shape[0] < 2 or n.shape[1] < 2:
        raise ValueError(f"panel nodes must be (n,>=2); got {n.shape}")
    seg = np.linalg.norm(np.diff(n[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _sample_arc_on_panel(
    sample: ContourForceSample,
    nodes: NDArray[np.floating],
    prefix: NDArray[np.float64],
) -> float | None:
    """Return arc-s of ``sample`` on the panel polyline, or None if off-strip."""
    p = np.array([float(sample.y), float(sample.z)], dtype=np.float64)
    n = np.asarray(nodes, dtype=np.float64)
    for i in range(n.shape[0] - 1):
        a = n[i, :2]
        b = n[i + 1, :2]
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 <= 0.0:
            if float(np.linalg.norm(p - a)) <= _COINCIDENCE_TOL_M:
                return float(prefix[i])
            continue
        t = float(np.dot(p - a, ab) / L2)
        t_cl = min(1.0, max(0.0, t))
        closest = a + t_cl * ab
        if float(np.linalg.norm(p - closest)) <= _COINCIDENCE_TOL_M:
            return float(prefix[i] + t_cl * np.sqrt(L2))
    return None


def sfv_to_n11_n12(
    sfv: SectionForceVector,
    panels: Sequence[object],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Interpolate SFV ``f_x``/``f_s`` onto each panel's stations as N11/N12 [N/m].

    Parameters
    ----------
    sfv
        Contour force samples (Channel C). Sample ``label`` must match the
        panel strip label (``skin:a`` / ``a`` forms both accepted).
    panels
        Objects with ``.nodes`` (n,2+), ``.s`` (n,), and ``.label``.

    Returns
    -------
    n11_list, n12_list
        One ``(n_station,)`` array per panel. Stations with no on-strip samples
        receive zeros — never values from another strip.
    """
    samples = list(sfv.samples)
    n11_list: list[NDArray[np.float64]] = []
    n12_list: list[NDArray[np.float64]] = []
    for panel in panels:
        nodes = np.asarray(getattr(panel, "nodes"), dtype=np.float64)
        s = np.asarray(getattr(panel, "s"), dtype=np.float64)
        label = str(getattr(panel, "label"))
        n_st = int(s.size)
        n11 = np.zeros(n_st, dtype=np.float64)
        n12 = np.zeros(n_st, dtype=np.float64)
        matched = _samples_for_panel_label(samples, label)
        if matched and n_st > 0 and nodes.shape[0] >= 2:
            prefix = _panel_arc_prefix(nodes)
            arcs: list[float] = []
            fx: list[float] = []
            fs: list[float] = []
            for samp in matched:
                arc = _sample_arc_on_panel(samp, nodes, prefix)
                if arc is None:
                    continue
                arcs.append(arc)
                fx.append(float(samp.f_x))
                fs.append(float(samp.f_s))
            if arcs:
                order = np.argsort(np.asarray(arcs, dtype=np.float64))
                a_ord = np.asarray(arcs, dtype=np.float64)[order]
                fx_ord = np.asarray(fx, dtype=np.float64)[order]
                fs_ord = np.asarray(fs, dtype=np.float64)[order]
                # Collapse duplicate arc stations (same point reported twice).
                uniq_a: list[float] = []
                uniq_fx: list[float] = []
                uniq_fs: list[float] = []
                for i in range(a_ord.size):
                    if uniq_a and abs(a_ord[i] - uniq_a[-1]) <= _COINCIDENCE_TOL_M:
                        uniq_fx[-1] = 0.5 * (uniq_fx[-1] + float(fx_ord[i]))
                        uniq_fs[-1] = 0.5 * (uniq_fs[-1] + float(fs_ord[i]))
                    else:
                        uniq_a.append(float(a_ord[i]))
                        uniq_fx.append(float(fx_ord[i]))
                        uniq_fs.append(float(fs_ord[i]))
                a_u = np.asarray(uniq_a, dtype=np.float64)
                if a_u.size == 1:
                    n11[:] = uniq_fx[0]
                    n12[:] = uniq_fs[0]
                else:
                    n11[:] = np.interp(s, a_u, np.asarray(uniq_fx, dtype=np.float64))
                    n12[:] = np.interp(s, a_u, np.asarray(uniq_fs, dtype=np.float64))
        n11_list.append(n11)
        n12_list.append(n12)
    return n11_list, n12_list


def sfv_to_nx_nxy(
    sfv: SectionForceVector,
    panels: Sequence[object],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Deprecated alias for :func:`sfv_to_n11_n12` (legacy Nx/Nxy names)."""
    return sfv_to_n11_n12(sfv, panels)


__all__ = ["sfv_to_n11_n12", "sfv_to_nx_nxy"]
