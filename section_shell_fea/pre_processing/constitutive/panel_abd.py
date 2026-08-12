"""Panel-level CLT ABD extraction shared by assembly and recovery paths."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def panel_abd_tuple_from_lam(lam: Any) -> tuple[np.ndarray, float, float, float, float]:
    """Return ``(ABD 6×6, thickness, nu, E, G_eff)`` for a panel laminate object."""
    stiffness_bundle = getattr(lam, "stiffness_bundle", None)
    if callable(stiffness_bundle):
        bundle = stiffness_bundle()
        ABD = np.asarray(bundle.ABD, dtype=np.float64).copy()
    elif callable(getattr(lam, "build_ABD", None)):
        ABD = np.asarray(lam.build_ABD(), dtype=np.float64).copy()
    else:
        from blade_structure_utilities.material_utilities.laminate import laminate_definition_from_legacy

        ld = laminate_definition_from_legacy(lam)
        if ld is None:
            raise TypeError(
                f"Panel laminate {type(lam).__name__!r} must expose stiffness_bundle(), "
                "build_ABD(), or convertible build_plies(); migrate to LaminateDefinition"
            )
        bundle = ld.stiffness_bundle()
        ABD = np.asarray(bundle.ABD, dtype=np.float64).copy()
    thickness = float(lam.t)
    nu = float(lam.nu)
    E_p = float(lam.E)
    G_eff = E_p / (2.0 * (1.0 + nu))
    return ABD, thickness, nu, E_p, G_eff


def compact_panel_abd_from_stacks(
    panel_maps: Sequence[Any],
    panel_abd_stacks: Sequence[tuple[np.ndarray, float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Convert per-panel-index ABD stacks to the compact recovery layout."""
    abds: list[np.ndarray] = []
    thicks: list[float] = []
    geffs: list[float] = []
    labels: list[str] = []
    for pi, pm in enumerate(panel_maps):
        if len(getattr(pm, "s_nodes", ())) == 0:
            continue
        ABD, thickness, _nu, _E_p, G_eff = panel_abd_stacks[pi]
        abds.append(np.asarray(ABD, dtype=np.float64))
        thicks.append(thickness)
        geffs.append(G_eff)
        labels.append(str(getattr(pm, "panel_label", "") or f"panel_{pi}"))
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
