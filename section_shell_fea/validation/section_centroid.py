"""Section geometry integrals used by FE handoffs (elastic centroid)."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def compute_elastic_centroid_from_panels(zones: list[Any]) -> tuple[float, float]:
    """A11-weighted elastic centroid (y_e, z_e) from panel laminates and geometry."""
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.constitutive.panel_abd import (
        panel_abd_tuple_from_lam,
    )

    sum_A11_y = 0.0
    sum_A11_z = 0.0
    sum_A11 = 0.0
    for p in zones:
        try:
            abd, _th, _nu, _E, _G = panel_abd_tuple_from_lam(p.lam)
            A11 = float(abd[0, 0])
        except Exception as exc:
            warnings.warn(
                "elastic-centroid: skipping panel "
                f"{getattr(p, 'label', '<unknown>')!r} — ABD assembly failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        s_p = np.asarray(p.s, dtype=float)
        if len(s_p) < 2 or len(nodes_yz) < 2:
            continue
        L_p = float(abs(s_p[-1] - s_p[0]))
        y_mid = float(np.mean(nodes_yz[:, 0]))
        z_mid = float(np.mean(nodes_yz[:, 1]))
        sum_A11_y += A11 * L_p * y_mid
        sum_A11_z += A11 * L_p * z_mid
        sum_A11 += A11 * L_p

    if sum_A11 < 1e-30:
        return 0.0, 0.0
    return float(sum_A11_y / sum_A11), float(sum_A11_z / sum_A11)
