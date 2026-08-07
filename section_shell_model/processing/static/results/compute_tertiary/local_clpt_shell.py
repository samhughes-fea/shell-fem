"""
Local CLPT shell subcomponent: full [N; M] solve from :class:`ShellPanelResultants`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import ShellPanelResultants


@dataclass
class StationCLPTShellResult:
    """Ply-level CLPT output at one station."""

    fi: np.ndarray
    eps0: np.ndarray
    kappa: np.ndarray
    sig_lam_mid: list[np.ndarray]
    eps_lam_mid: list[np.ndarray]
    plies: Any
    N_vec: np.ndarray
    M_vec: np.ndarray


def solve_station_clpt_shell(
    resultants: ShellPanelResultants,
    plies: list | Any,
    *,
    Xt: float | NDArray[np.float64] | None = None,
    Xc: float | NDArray[np.float64] | None = None,
    Yt: float | NDArray[np.float64] | None = None,
    Yc: float | NDArray[np.float64] | None = None,
    S12: float | NDArray[np.float64] | None = None,
    lam: Any | None = None,
) -> StationCLPTShellResult:
    """
    Solve CLPT for ``[N; M] = [[A,B],[B,D]] [eps0; kappa]`` and Hashin envelope FI.

    Ply stresses from CLPT recovery are passed to
    :func:`blade_structure_utilities.material_utilities.failure.hashin_fi` with per-ply
    strength arrays when scalars are broadcast.
    """
    from blade_structure_utilities.material_utilities.laminate import clpt_recovery_from_laminate
    from blade_structure_utilities.material_utilities.failure import hashin_fi
    from blade_structure_utilities.material_utilities.laminate import LaminateDefinition, laminate_definition_from_legacy

    N_vec = resultants.to_N_vec()
    M_vec = resultants.to_M_vec()

    laminate = laminate_definition_from_legacy(lam, plies=plies if isinstance(plies, list) else None)
    if laminate is None or not isinstance(laminate, LaminateDefinition):
        raise TypeError(
            "solve_station_clpt_shell requires LaminateDefinition via lam= or convertible plies"
        )

    n_ply = len(laminate.plies)
    _def = 1e9
    Xt_a = np.full(n_ply, _def, dtype=float)
    Xc_a = np.full(n_ply, _def, dtype=float)
    Yt_a = np.full(n_ply, _def, dtype=float)
    Yc_a = np.full(n_ply, _def, dtype=float)
    S12_a = np.full(n_ply, _def, dtype=float)
    for arr, val in (
        (Xt_a, Xt), (Xc_a, Xc), (Yt_a, Yt), (Yc_a, Yc), (S12_a, S12),
    ):
        if val is None:
            continue
        if isinstance(val, np.ndarray):
            v = np.asarray(val, dtype=float).ravel()
            arr[: min(n_ply, v.size)] = v[: min(n_ply, v.size)]
        else:
            arr[:] = float(val)

    eps0, kappa, sig_lam, eps_lam = clpt_recovery_from_laminate(laminate, N_vec, M_vec)

    fi = np.zeros(n_ply, dtype=float)
    for i, sig in enumerate(sig_lam):
        if i >= n_ply:
            break
        sig3 = np.asarray(sig, dtype=float).ravel()[:3]
        fi[i] = float(
            hashin_fi(
                sig3.reshape(1, 3),
                Xt_a[i : i + 1], Xc_a[i : i + 1], Yt_a[i : i + 1],
                Yc_a[i : i + 1], S12_a[i : i + 1],
            )[0]
        )

    return StationCLPTShellResult(
        fi=fi,
        eps0=eps0,
        kappa=kappa,
        sig_lam_mid=list(sig_lam),
        eps_lam_mid=list(eps_lam),
        plies=laminate.build_clpt_ply_rows(),
        N_vec=N_vec,
        M_vec=M_vec,
    )


def sweep_panel_clpt_fi(
    panel_resultants: list,
    plies: list | Any,
    *,
    Xt: float,
    Xc: float,
    Yt: float,
    Yc: float,
    S12: float,
    lam: Any | None = None,
) -> list[StationCLPTShellResult]:
    """Run CLPT Hashin-envelope failure analysis on every element in ``panel_resultants``."""
    return [
        solve_station_clpt_shell(
            r, plies, Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12, lam=lam,
        )
        for r in panel_resultants
    ]


def default_skin_strengths_pa() -> dict[str, float]:
    """Representative glass/carbon skin strengths [Pa] for Hashin envelope demos."""
    return {
        "Xt": 600e6,
        "Xc": 500e6,
        "Yt": 50e6,
        "Yc": 140e6,
        "S12": 45e6,
    }
