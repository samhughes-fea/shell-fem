"""Section geometry integrals used by FE handoffs (elastic centroid)."""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_elastic_centroid_from_panels(zones: list[Any]) -> tuple[float, float]:
    """A11-weighted elastic centroid (y_e, z_e) from panel laminates and geometry.

    Every panel with usable geometry must contribute. ABD assembly failure or an
    empty A11 sum raises (FEA-C-001) — never skip panels or return a silent
    ``(0, 0)`` pierce.
    """
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.constitutive.panel_abd import (
        panel_abd_tuple_from_lam,
    )

    sum_A11_y = 0.0
    sum_A11_z = 0.0
    sum_A11 = 0.0
    for p in zones:
        label = getattr(p, "label", "<unknown>")
        try:
            abd, _th, _nu, _E, _G = panel_abd_tuple_from_lam(p.lam)
            A11 = float(abd[0, 0])
        except Exception as exc:
            raise ValueError(
                f"elastic-centroid: ABD assembly failed for panel {label!r} "
                f"(FEA-C-001): {exc}"
            ) from exc
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        s_p = np.asarray(p.s, dtype=float)
        if len(s_p) < 2 or len(nodes_yz) < 2:
            raise ValueError(
                f"elastic-centroid: panel {label!r} needs ≥2 stations/nodes "
                f"(got len(s)={len(s_p)}, len(nodes)={len(nodes_yz)}; FEA-C-001)"
            )
        L_p = float(abs(s_p[-1] - s_p[0]))
        y_mid = float(np.mean(nodes_yz[:, 0]))
        z_mid = float(np.mean(nodes_yz[:, 1]))
        sum_A11_y += A11 * L_p * y_mid
        sum_A11_z += A11 * L_p * z_mid
        sum_A11 += A11 * L_p

    if sum_A11 < 1e-30:
        raise ValueError(
            "elastic-centroid: A11-weighted length sum is ~0; refusing silent "
            "(0, 0) pierce (FEA-C-001)"
        )
    return float(sum_A11_y / sum_A11), float(sum_A11_z / sum_A11)
