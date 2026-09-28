"""
Station CLPT shell subcomponent: full [N; M] solve from :class:`ShellPanelResultants`.

Frames (see ``geometry_utilities.coordinate_frames``): the CLPT solve and its
``sig_lam_mid`` / ``eps_lam_mid`` output are in the element/laminate frame **E**;
``sig_ply_mat_mid`` / ``eps_ply_mat_mid`` are the same quantities rotated into the
ply material frame **M**. Hashin is evaluated on **M** only — feeding E-frame
stress to ``hashin_fi`` under-predicts the index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellPanelResultants
from blade_structure_utilities.geometry_utilities.coordinate_frames import (
    FRAME_E,
    FRAME_M,
    FrameTag,
    require_frame_tag,
)


@dataclass
class StationCLPTShellResult:
    """Ply-level CLPT output at one station.

    ``sig_lam_mid`` / ``eps_lam_mid`` are in **E**; ``sig_ply_mat_mid`` /
    ``eps_ply_mat_mid`` are in **M**. ``fi`` is evaluated on the M-frame stress.
    """

    fi: np.ndarray
    eps0: np.ndarray
    kappa: np.ndarray
    sig_lam_mid: list[np.ndarray]
    eps_lam_mid: list[np.ndarray]
    plies: Any
    N_vec: np.ndarray
    M_vec: np.ndarray
    sig_ply_mat_mid: list[np.ndarray] | None = None
    eps_ply_mat_mid: list[np.ndarray] | None = None
    mid_surface_frame: FrameTag = FRAME_E
    ply_material_frame: FrameTag = FRAME_M

    def __post_init__(self) -> None:
        self.mid_surface_frame = require_frame_tag(self.mid_surface_frame)
        self.ply_material_frame = require_frame_tag(self.ply_material_frame)
        if self.mid_surface_frame != FRAME_E:
            raise ValueError(
                f"StationCLPTShellResult.mid_surface_frame must be {FRAME_E!r}; "
                f"got {self.mid_surface_frame!r} (FEA-R-003)"
            )
        if self.ply_material_frame != FRAME_M:
            raise ValueError(
                f"StationCLPTShellResult.ply_material_frame must be {FRAME_M!r}; "
                f"got {self.ply_material_frame!r} (FEA-R-003)"
            )


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

    CLPT recovery returns ply stress/strain in **E**; both are rotated into **M**
    (stress by :math:`T_\\sigma`, engineering strain by :math:`T_\\varepsilon` --
    different matrices) before
    :func:`blade_structure_utilities.material_utilities.failure.hashin_fi` sees
    them, with per-ply strength arrays when scalars are broadcast.
    """
    from blade_structure_utilities.material_utilities.laminate import clpt_recovery_from_laminate
    from blade_structure_utilities.material_utilities.failure import hashin_fi
    from blade_structure_utilities.material_utilities.laminate import (
        LaminateDefinition,
        laminate_definition_from_legacy,
        rotate_plies_to_material,
    )

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

    # E → M. Stress takes T_sigma and engineering strain takes T_epsilon; they are
    # *different* matrices, and both are owned by coordinate_frames (the memoised
    # bundle entries delegate to ply_stress_transform_E_to_M / _strain_).
    bundle = laminate.stiffness_bundle()
    sig_lam_arr = np.asarray(sig_lam, dtype=np.float64).reshape(-1, 3)
    eps_lam_arr = np.asarray(eps_lam, dtype=np.float64).reshape(-1, 3)
    T_sig = np.asarray(bundle.T_ply, dtype=np.float64)
    T_eps = np.asarray(bundle.T_ply_eps, dtype=np.float64)
    sig_mat = rotate_plies_to_material(sig_lam_arr[None, ...], T_sig[None, ...])[0]
    eps_mat = rotate_plies_to_material(eps_lam_arr[None, ...], T_eps[None, ...])[0]

    fi = np.zeros(n_ply, dtype=float)
    for i in range(min(n_ply, sig_mat.shape[0])):
        sig3 = sig_mat[i].ravel()[:3]
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
        sig_ply_mat_mid=[np.asarray(r, dtype=float) for r in sig_mat],
        eps_ply_mat_mid=[np.asarray(r, dtype=float) for r in eps_mat],
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
    """Representative glass/carbon skin strengths [Pa] for Hashin **demos only**.

    Production governing FI must use laminate ply strengths (FEA-FI-001). Do not
    call this as a silent substitute when a zone laminate is available.
    """
    return {
        "Xt": 600e6,
        "Xc": 500e6,
        "Yt": 50e6,
        "Yc": 140e6,
        "S12": 45e6,
    }
