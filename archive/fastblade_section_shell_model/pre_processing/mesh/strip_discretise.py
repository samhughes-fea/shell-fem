"""Strip mesh discretisation — shared by mesh builder and global solvers.

Delegates 1-D arc-length node placement to :mod:`blade_utilities.mesh.line_1d`
and spanwise quad connectivity to :mod:`blade_utilities.mesh.line_1d`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from blade_utilities.mesh.line_1d import LineMeshSpec, strip_quad_connectivity


def _panel_mesh_kind(panel_label: str) -> str:
    """Return the kind prefix (``"cap"``, ``"web"``, ``"skin"``) from a panel label."""
    if ":" in panel_label:
        return str(panel_label.split(":", 1)[0])
    return ""


def discretise_strip(
    s_panel: NDArray[np.float64],
    n_elements: int,
    *,
    panel_label: str = "",
    nodes_yz: NDArray[np.float64] | None = None,
    line_spec: LineMeshSpec | None = None,
) -> tuple[NDArray[np.float64], list[list[int]]]:
    """Equi-arc strip nodes via ``line_1d`` policies, with knot-preserving caps.

    Parameters
    ----------
    s_panel
        Cumulative arc-length coordinates of the panel midline polyline nodes.
    n_elements
        Requested element count (skin/web; caps use knot-preserving policy).
    panel_label
        ``"<kind>:<name>"`` — ``kind`` selects the ``line_1d`` policy.
    nodes_yz
        Accepted for API compatibility with the solver (unused here).
    line_spec
        Optional override for :class:`LineMeshSpec`.
    """
    del nodes_yz
    s_panel = np.asarray(s_panel, dtype=float).ravel()
    kind = _panel_mesh_kind(panel_label)
    spec = line_spec if line_spec is not None else LineMeshSpec()
    policy = spec.policy_for_kind(kind or "panel")
    s_nodes = policy.s_nodes(
        s_panel,
        n_segments=max(1, int(n_elements)),
        spec=spec,
    )
    n = len(s_nodes)
    return np.asarray(s_nodes, dtype=float), strip_quad_connectivity(n)


__all__ = ["discretise_strip", "strip_quad_connectivity"]
