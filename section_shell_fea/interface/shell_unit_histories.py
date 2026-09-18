"""Reshape shell tertiary sig_ply_mat into fatigue unit_histories dicts.

FE-free consumers (fatigue_branch ShellElementStressRecovery) take plain
dicts. This adapter owns the (n_elem, n_ply, 3, 3) layout contract.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np
from numpy.typing import NDArray

FACE_NAMES = ("OUTER", "MID", "INNER")
FACE_OUTER, FACE_MID, FACE_INNER = 0, 1, 2


def _validate_sig_ply_mat(sig_ply_mat: NDArray[np.floating]) -> NDArray[np.float64]:
    sig = np.asarray(sig_ply_mat, dtype=np.float64)
    if sig.ndim != 4:
        raise ValueError(f"sig_ply_mat must be 4-D (n_elem, n_ply, 3, 3), got {sig.shape}")
    if sig.shape[2] != 3 or sig.shape[3] != 3:
        raise ValueError(
            f"sig_ply_mat face/component axes must be 3, got shape {sig.shape}"
        )
    if sig.shape[0] < 1 or sig.shape[1] < 1:
        raise ValueError(f"sig_ply_mat must have at least one element and ply, got {sig.shape}")
    return sig


def isotropic_unit_histories_from_sig_ply_mat(
    sig_ply_mat: NDArray[np.floating],
    *,
    ply_index: int = 0,
    element_index: int = 0,
) -> Dict[str, Dict[str, NDArray[np.float64]]]:
    """One isotropic element -> OUTER/MID/INNER component dicts.

    Uses ply ``ply_index`` (monolayer = 0). Arrays are length-1 (static) unless
    the caller later replaces them with time series of the same layout.
    """
    sig = _validate_sig_ply_mat(sig_ply_mat)
    row = sig[int(element_index), int(ply_index)]  # (3, 3)
    out: Dict[str, Dict[str, NDArray[np.float64]]] = {}
    for face_i, name in enumerate(FACE_NAMES):
        s11, s22, s12 = row[face_i]
        out[name] = {
            "s11": np.asarray([s11], dtype=np.float64),
            "s22": np.asarray([s22], dtype=np.float64),
            "s12": np.asarray([s12], dtype=np.float64),
        }
    return out


def composite_unit_histories_from_sig_ply_mat(
    sig_ply_mat: NDArray[np.floating],
    *,
    face: int = FACE_MID,
    element_index: int = 0,
    n_ply: Optional[int] = None,
) -> Dict[str, Dict[str, NDArray[np.float64]]]:
    """One composite element -> ply_k component dicts at a single face (default MID)."""
    sig = _validate_sig_ply_mat(sig_ply_mat)
    n_ply_use = int(sig.shape[1] if n_ply is None else n_ply)
    if n_ply_use < 1 or n_ply_use > sig.shape[1]:
        raise ValueError(f"n_ply={n_ply_use} incompatible with shape {sig.shape}")
    if not (0 <= int(face) < 3):
        raise ValueError(f"face must be 0..2, got {face}")
    out: Dict[str, Dict[str, NDArray[np.float64]]] = {}
    for ply in range(n_ply_use):
        s11, s22, t12 = sig[int(element_index), ply, int(face)]
        out[f"ply_{ply}"] = {
            "sigma_1": np.asarray([s11], dtype=np.float64),
            "sigma_2": np.asarray([s22], dtype=np.float64),
            "tau_12": np.asarray([t12], dtype=np.float64),
        }
    return out


def unit_histories_from_sig_ply_mat(
    sig_ply_mat: NDArray[np.floating],
    *,
    kind: str,
    subcomponent: str = "shell",
    element_index: int = 0,
    face: int = FACE_MID,
    ply_index: int = 0,
    n_ply: Optional[int] = None,
) -> Dict[str, Any]:
    """Build ``{subcomponent: histories}`` for ShellElementStressRecovery.

    Parameters
    ----------
    kind
        ``isotropic`` (three faces) or ``composite`` (per-ply at ``face``).
    """
    kind_l = str(kind).lower()
    if kind_l == "isotropic":
        hist = isotropic_unit_histories_from_sig_ply_mat(
            sig_ply_mat, ply_index=ply_index, element_index=element_index
        )
    elif kind_l == "composite":
        hist = composite_unit_histories_from_sig_ply_mat(
            sig_ply_mat, face=face, element_index=element_index, n_ply=n_ply
        )
    else:
        raise ValueError(f"kind must be isotropic or composite, got {kind!r}")
    return {subcomponent: hist}


def merge_unit_histories(
    *parts: Mapping[str, Any],
) -> Dict[str, Any]:
    """Shallow-merge subcomponent -> history maps (later wins on key clash)."""
    out: Dict[str, Any] = {}
    for part in parts:
        out.update(dict(part))
    return out


__all__ = [
    "FACE_NAMES",
    "FACE_OUTER",
    "FACE_MID",
    "FACE_INNER",
    "isotropic_unit_histories_from_sig_ply_mat",
    "composite_unit_histories_from_sig_ply_mat",
    "unit_histories_from_sig_ply_mat",
    "merge_unit_histories",
]
