"""Consistent nodal loads from a ``SectionForceVector`` (FEA-L-001).

Each strip's contour force field is integrated against **that strip's own**
element shape functions:

* the field of a strip is rebuilt from its samples -- Channel C emits every
  segment's start / mid / end with Simpson weights, i.e. an exact quadratic per
  segment; a producer without that structure (Channels A and B) is read as a
  linear field through its samples;
* every sample must lie on its strip's polyline (to 1 nm) -- its arc position is
  located, not guessed; a sample off its strip is refused;
* each linear element ``[s_i, s_{i+1}]`` is split at the field's breakpoints and
  ``∫ f N ds`` is integrated exactly on every piece (Simpson: the integrand is at
  most cubic);
* the element loads go 50/50 to the node's bottom/top pair, and the in-plane
  force vector is resolved onto each node's own ``(t, n = normal_2d(t))`` -- the
  convention ``applied_force_resultant_yz`` inverts.

Because the hat functions sum to one, the applied force equals the field's force
exactly. This replaced lumping each sample's whole force, with position-derived
weights, onto the nearest node pair of any strip.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.load_utilities.section_loads import SectionForceVector
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.panel_node_assembly import (
    normal_2d,
    strip_identity_at_node,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.analysis_dof_layout import (
    U_S,
    U_X,
    W,
    gdof,
)

__all__ = ["section_force_vector_to_nodal_load"]

_ON_STRIP_TOL_M = 1e-9


def _strip_label(panel_label: str) -> str:
    """``kind:label`` → ``label`` (panel labels carry the strip kind as a prefix)."""
    kind, _, rest = str(panel_label).partition(":")
    return rest if kind in ("skin", "web", "cap") and rest else str(panel_label)


def _labels_match(a: str, b: str) -> bool:
    return a == b or _strip_label(a) == _strip_label(b)


def _require_interior_handoff_membership(node_meta: Any, pm: Any) -> None:
    """Interior stations must carry the producer strip label (SM-002 / FEA-L-001).

    Endpoints may be shared junctions (first-writer identity from another strip);
    interiors must match this panel's label with no nearest-YZ rediscovery.
    """
    labels = getattr(node_meta, "strip_label", None)
    if labels is None:
        return
    # Skip when the handoff was never stamped (legacy unit fixtures).
    if not any(str(x) for x in labels):
        return
    s_nodes = np.asarray(pm.s_nodes, dtype=float)
    n = int(s_nodes.size)
    if n < 3:
        return
    expected = str(pm.panel_label)
    for i in range(1, n - 1):
        for gn in (pm.global_nodes[i], pm.global_nodes[i + n]):
            if int(gn) < 0:
                continue
            got, _kind, _arc = strip_identity_at_node(node_meta, int(gn))
            if not _labels_match(got, expected):
                raise ValueError(
                    f"global node {gn} handoff strip {got!r} != panel "
                    f"{expected!r}; refusing nearest-YZ membership rediscovery "
                    "(SM-002)"
                )


def _arc_position(points: NDArray[np.float64], prefix: NDArray[np.float64], target, *, label: str) -> float:
    a, b = points[:-1], points[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", target - a, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
    distance = np.hypot(*(a + t[:, None] * ab - target).T)
    segment = int(np.argmin(distance))
    if float(distance[segment]) > _ON_STRIP_TOL_M:
        raise ValueError(
            f"a {label!r} load sample lies {float(distance[segment]):.3e} m off its strip; samples "
            "must lie on the strip they belong to"
        )
    return float(prefix[segment] + t[segment] * np.hypot(*ab[segment]))


def _field_pieces(samples: Sequence[Any], points: NDArray[np.float64], prefix: NDArray[np.float64], label: str):
    """``[(s0, s1, values(3, 3))]``: quadratic pieces, values = (f_x, F_X, F_Y) at s0, mid, s1."""
    values = np.array(
        [
            [s.f_x, s.f_s * s.t_y + s.f_w * s.n_y, s.f_s * s.t_z + s.f_w * s.n_z]
            for s in samples
        ],
        dtype=np.float64,
    )
    arc = np.array([_arc_position(points, prefix, np.array([s.y, s.z]), label=label) for s in samples])
    weights = np.array([float(getattr(s, "ds", 0.0) or 0.0) for s in samples])
    pieces = []
    triplets = len(samples) % 3 == 0 and np.all(weights > 0.0)
    if triplets:
        w = weights.reshape(-1, 3)
        triplets = bool(np.allclose(w[:, 1], 4.0 * w[:, 0], rtol=1e-9) and np.allclose(w[:, 0], w[:, 2], rtol=1e-9))
    if triplets:
        for start in range(0, len(samples), 3):
            s0, s1 = arc[start], arc[start + 2]
            block = values[start : start + 3]
            if s1 < s0:
                s0, s1, block = s1, s0, block[::-1]
            pieces.append((s0, s1, block))
    else:
        order = np.argsort(arc)
        for i, j in zip(order[:-1], order[1:]):
            if arc[j] > arc[i]:
                pieces.append((arc[i], arc[j], np.stack([values[i], 0.5 * (values[i] + values[j]), values[j]])))
    return pieces


def _quadratic(block: NDArray[np.float64], s0: float, s1: float, s: float) -> NDArray[np.float64]:
    """The quadratic through ``block`` at ``(s0, mid, s1)``, evaluated at ``s``."""
    xi = (s - s0) / (s1 - s0)
    return block[0] * (1 - xi) * (1 - 2 * xi) + block[1] * 4 * xi * (1 - xi) + block[2] * xi * (2 * xi - 1)


def section_force_vector_to_nodal_load(
    node_meta: Any,
    sfv: SectionForceVector,
    *,
    panel_maps: Sequence[Any],
    zones: Sequence[Any],
    ndof: int = 6,
    L_x: float = 1.0,
    f: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Consistent nodal load vector for a contour force field.

    Parameters
    ----------
    node_meta
        ``SectionNodeAssemblyMetadata`` (node tangents define each node's ``(u_s, w)``).
    sfv
        The contour force field; sample labels are strip labels.
    panel_maps, zones
        The assembly's ``PanelNodeAssemblyMap`` list and the zones it was built from
        (aligned): each panel's ``s_nodes``, bottom/top ``global_nodes`` and polyline.
    L_x
        Station length [m] the densities are integrated over.

    Raises
    ------
    ValueError
        A sample off its strip; samples for a strip that has no panel.
    """
    n_gdof = int(node_meta.n_nodes) * int(ndof)
    if f is None:
        f = np.zeros(n_gdof, dtype=float)
    else:
        f = np.asarray(f, dtype=float)
        if f.size != n_gdof:
            raise ValueError(f"f length {f.size} != n_gdof {n_gdof}")
    if len(panel_maps) != len(zones):
        raise ValueError("panel_maps and zones must be aligned")

    by_label: dict[str, list[Any]] = {}
    for sample in sfv.samples:
        by_label.setdefault(str(sample.label), []).append(sample)
    panel_of = {_strip_label(pm.panel_label): index for index, pm in enumerate(panel_maps)}
    orphans = sorted(set(by_label) - set(panel_of))
    if orphans:
        raise ValueError(f"load samples for strips {orphans} have no FE panel")

    tangents = np.asarray(node_meta.tangent_yz, dtype=float)
    scale = float(L_x)
    for label, samples in by_label.items():
        index = panel_of[label]
        pm, zone = panel_maps[index], zones[index]
        s_nodes = np.asarray(pm.s_nodes, dtype=float)
        if s_nodes.size < 2:
            continue
        _require_interior_handoff_membership(node_meta, pm)
        points = np.asarray(zone.nodes, dtype=float)
        prefix = np.asarray(zone.s, dtype=float)
        pieces = _field_pieces(samples, points, prefix, label)
        nodal = np.zeros((s_nodes.size, 3))
        for element in range(s_nodes.size - 1):
            a, b = s_nodes[element], s_nodes[element + 1]
            for s0, s1, block in pieces:
                lo, hi = max(a, s0), min(b, s1)
                if hi <= lo:
                    continue
                for s, weight in ((lo, 1.0), (0.5 * (lo + hi), 4.0), (hi, 1.0)):
                    value = _quadratic(block, s0, s1, s) * weight * (hi - lo) / 6.0
                    shape_b = (s - a) / (b - a)
                    nodal[element] += (1.0 - shape_b) * value
                    nodal[element + 1] += shape_b * value
        count = s_nodes.size
        for i in range(count):
            f_axial, force_x, force_y = nodal[i] * scale
            for gn in (pm.global_nodes[i], pm.global_nodes[i + count]):
                t = tangents[gn] / float(np.hypot(*tangents[gn]))
                n = normal_2d(t)
                f[gdof(gn, U_X, ndof)] += 0.5 * f_axial
                f[gdof(gn, U_S, ndof)] += 0.5 * (force_x * t[0] + force_y * t[1])
                f[gdof(gn, W, ndof)] += 0.5 * (force_x * n[0] + force_y * n[1])
    return f
