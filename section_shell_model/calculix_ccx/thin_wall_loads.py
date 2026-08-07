"""Vlasov warping + per-panel Nx/Nxy load prep for CCX thin-wall validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.thin_wall.section_vlasov import (
    SectionVlasovResult,
    compute_section_vlasov,
)


def prepare_thin_wall_panel_loads(
    airfoil: np.ndarray,
    panels: list,
    webs_geom: Any,
    q_tot: list,
    sig_p: list,
    B: float,
    dB_dx: float,
) -> tuple[SectionVlasovResult, list[np.ndarray], list[np.ndarray]]:
    """Vlasov warping + per-panel ``Nx``/``Nxy`` nodal load arrays for global coupled solvers."""
    t_web_mean = float(np.mean([p.lam.t for p in panels if len(p.s) >= 2])) if panels else None
    vlasov = compute_section_vlasov(
        airfoil, panels, B, dB_dx,
        webs_geom=list(webs_geom),
        t_web=t_web_mean,
    )

    nx_panels: list[np.ndarray] = []
    nxy_panels: list[np.ndarray] = []
    for pi, p in enumerate(panels):
        if len(p.s) < 2:
            nx_panels.append(np.array([]))
            nxy_panels.append(np.array([]))
            continue
        thickness = float(p.lam.t)
        sig_xx_arr = np.asarray(sig_p[pi], dtype=float)
        is_web = "web" in str(getattr(p, "label", "")).lower()
        if is_web:
            sig_w_nodes = np.zeros_like(sig_xx_arr)
        elif pi < len(vlasov.sigma_omega) and len(vlasov.sigma_omega[pi]) > 0:
            s_mids = vlasov.panel_s_mids[pi]
            sig_w_mids = vlasov.sigma_omega[pi]
            if len(s_mids) > 0 and len(p.s) >= 2:
                sig_w_nodes = np.interp(
                    p.s, np.r_[0.0, s_mids, p.s[-1]], np.r_[sig_w_mids[0], sig_w_mids, sig_w_mids[-1]]
                )
            else:
                sig_w_nodes = np.zeros_like(sig_xx_arr)
        else:
            sig_w_nodes = np.zeros_like(sig_xx_arr)
        sig_total = sig_xx_arr + sig_w_nodes
        nx_panels.append(sig_total * thickness)
        nxy_panels.append(np.asarray(q_tot[pi], dtype=float))

    return vlasov, nx_panels, nxy_panels


__all__ = ["prepare_thin_wall_panel_loads"]
