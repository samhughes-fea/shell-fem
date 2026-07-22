"""MITC4 solve orchestration: ABD collection and thin-wall load prep for global-coupled MITC4.

Helpers used by the periodic section solve
(:mod:`~blade_precompute.section_shell_model.api.periodic`) and by assembly /
CCX validation adapters. The former clamped entry
(``api/legacy_clamped.py`` / ``run_section_with_mitc4_shell``) has been removed.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from blade_precompute.section_shell_model.engine.section_vlasov import (
    SectionVlasovResult,
    compute_section_vlasov,
)


def panel_abd_tuple_from_lam(lam: Any) -> tuple[np.ndarray, float, float, float, float]:
    """Return ``(ABD 6×6, thickness, nu, E, G_eff)`` for a panel laminate object."""
    from blade_precompute.section_shell_model.pre_processing.constitutive.panel_abd import (
        panel_abd_tuple_from_lam as _panel_abd_tuple_from_lam,
    )

    return _panel_abd_tuple_from_lam(lam)


def compact_panel_abd_from_stacks(
    panel_maps: Sequence[Any],
    panel_abd_stacks: Sequence[tuple[np.ndarray, float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Convert per-panel-index ABD stacks to the compact ``collect_panel_abd_stack`` layout."""
    from blade_precompute.section_shell_model.pre_processing.constitutive.panel_abd import (
        compact_panel_abd_from_stacks as _compact,
    )

    return _compact(panel_maps, panel_abd_stacks)


def collect_panel_abd_stack(panels: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build per-panel CLT stiffness ``ABD`` (6×6) and scalars used by shell assembly.

    Returns
    -------
    abd_stack : (n_valid, 6, 6)
    thickness_m : (n_valid,)
    G_eff : (n_valid,)
    labels : length n_valid
    """
    abds: list[np.ndarray] = []
    thicks: list[float] = []
    geffs: list[float] = []
    labels: list[str] = []
    for pi, p in enumerate(panels):
        s = getattr(p, "s", None)
        if s is None or len(s) < 2:
            continue
        ABD, thickness, _nu, _E_p, G_eff = panel_abd_tuple_from_lam(p.lam)
        abds.append(np.asarray(ABD, dtype=np.float64))
        thicks.append(thickness)
        geffs.append(G_eff)
        labels.append(str(getattr(p, "label", "") or f"panel_{pi}"))
    if not abds:
        return (
            np.zeros((0, 6, 6), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            [],
        )
    return (
        np.stack(abds, axis=0),
        np.asarray(thicks, dtype=np.float64),
        np.asarray(geffs, dtype=np.float64),
        labels,
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


__all__ = [
    "collect_panel_abd_stack",
    "prepare_thin_wall_panel_loads",
]
