"""Fringe plots of element results on a shell model (matplotlib, installed with ``[plots]``).

``surface_fringe`` draws every element of a set as a polygon projected onto two global
axes and coloured by one value per element: the span-chord view (z, x) of a blade skin
surface is the default, so a suction or pressure side reads as a planform map.
``section_fringe`` draws the elements of one span bay as segments in the section plane
(x, y), coloured the same way: skins, caps and webs of one cross-section. Both take
values as an array aligned with ``element_ids`` (for example a ``PlyFailureField``'s
``max_index`` and ``element_ids``), return the matplotlib collection for a colour bar,
and leave the rest of the axes to the caller.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def _matplotlib():
    try:
        from matplotlib.collections import LineCollection, PolyCollection
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("fringe plots need matplotlib: pip install shell-fem[plots]") from exc
    return LineCollection, PolyCollection


def _lookup(element_ids, values):
    ids = np.asarray(element_ids, dtype=int).ravel()
    v = np.asarray(values, dtype=float).ravel()
    if ids.size != v.size:
        raise ValueError("values need one entry per element id")
    return dict(zip(ids.tolist(), v.tolist()))


def _corners(mesh, element) -> np.ndarray:
    n = getattr(element, "N_CORNERS", 0) or len(element.node_ids)
    return np.array([mesh.node_coordinates(nid) for nid in element.node_ids[:n]])


def surface_fringe(ax, mesh, element_ids: Sequence[int], values: Sequence[float],
                   show: Optional[Sequence[int]] = None, axes: Tuple[int, int] = (2, 0), cmap="viridis",
                   vmin: Optional[float] = None, vmax: Optional[float] = None, edgecolor="none", **kwargs):
    """Elements ``show`` (all of ``element_ids`` by default) as polygons on the global ``axes``, coloured by value.

    ``axes`` = (2, 0) puts span z across and chord x up. Elements without a value are
    skipped. Returns the ``PolyCollection``.
    """
    _, PolyCollection = _matplotlib()
    lookup = _lookup(element_ids, values)
    show = list(lookup) if show is None else [int(e) for e in show]
    polygons, colours = [], []
    for eid in show:
        if eid not in lookup or not np.isfinite(lookup[eid]):
            continue
        polygons.append(_corners(mesh, mesh.element(eid))[:, list(axes)])
        colours.append(lookup[eid])
    collection = PolyCollection(polygons, array=np.asarray(colours), cmap=cmap, edgecolors=edgecolor, **kwargs)
    collection.set_clim(vmin if vmin is not None else (min(colours) if colours else 0.0),
                        vmax if vmax is not None else (max(colours) if colours else 1.0))
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def section_fringe(ax, mesh, element_ids: Sequence[int], values: Sequence[float], z: float,
                   show: Optional[Sequence[int]] = None, cmap="viridis", vmin: Optional[float] = None,
                   vmax: Optional[float] = None, linewidth: float = 4.0, **kwargs):
    """The elements crossing span position ``z`` as segments in the (x, y) section plane, coloured by value.

    Each element whose span extent holds ``z`` is drawn as the chord of its corners
    nearest ``z`` (the cut through a quad's span edges, a triangle's nearest edge).
    Elements lying in the plane (ribs) are skipped. Returns the ``LineCollection``.
    """
    LineCollection, _ = _matplotlib()
    lookup = _lookup(element_ids, values)
    show = list(lookup) if show is None else [int(e) for e in show]
    segments, colours = [], []
    for eid in show:
        if eid not in lookup or not np.isfinite(lookup[eid]):
            continue
        xyz = _corners(mesh, mesh.element(eid))
        z_lo, z_hi = xyz[:, 2].min(), xyz[:, 2].max()
        if not z_lo <= z <= z_hi or z_hi - z_lo <= 1e-9 * max(1.0, abs(z)):
            continue
        points = []
        for a, b in zip(xyz, np.roll(xyz, -1, axis=0)):    # where each edge crosses the plane z
            if (a[2] - z) * (b[2] - z) <= 0.0 and a[2] != b[2]:
                t = (z - a[2]) / (b[2] - a[2])
                points.append((a + t * (b - a))[:2])
        if len(points) >= 2:
            points = np.array(points)
            far = np.argmax(np.linalg.norm(points - points[0], axis=1))
            segments.append([points[0], points[far]])
            colours.append(lookup[eid])
    collection = LineCollection(segments, array=np.asarray(colours), cmap=cmap, linewidths=linewidth, **kwargs)
    collection.set_clim(vmin if vmin is not None else (min(colours) if colours else 0.0),
                        vmax if vmax is not None else (max(colours) if colours else 1.0))
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    return collection


__all__ = ["section_fringe", "surface_fringe"]
