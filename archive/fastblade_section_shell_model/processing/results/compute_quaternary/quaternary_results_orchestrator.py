"""Tier 4 orchestrator: failure indices from tier 3 ply-resolved CLT state.

Pure consumer of ``ShellTertiaryResultSet`` (+ the formulation cache, for
strength-vector lookups via ``cache.panel_laminates`` and constitutive routing
via ``cache.panel_constitutive_kind`` — mirroring tier 3's laminate resolution).
See PROCESSING_ARCHITECTURE_PROPOSAL.md tier 4 and docs/TAXONOMY.md.
"""

from __future__ import annotations

import math

import numpy as np

from blade_precompute.section_shell_model.pre_processing.containers.formulation_objects import ShellFormulationCache
from ..containers.container_hopper import ShellHashinFiResult, ShellQuaternaryResultSet, ShellTertiaryResultSet
from .compute_hashin_fi import (
    batched_interlaminar_fi,
    interlaminar_allowable_from_lam,
    interlaminar_fi_from_interface_tau,
    ply_hashin_fi,
    ply_strength_vectors_from_lam,
)
from .compute_von_mises_fi import von_mises_fi


def _panel_kind(cache: ShellFormulationCache, pi: int) -> str:
    kinds = cache.panel_constitutive_kind
    if pi < len(kinds) and kinds[pi]:
        return str(kinds[pi])
    return "orthotropic"


def _panel_sigma_allow(cache: ShellFormulationCache, pi: int) -> float:
    allows = cache.panel_sigma_allow_pa
    if pi < len(allows):
        return float(allows[pi])
    return float("nan")


class ShellQuaternaryResultsOrchestrator:
    """Hashin (orthotropic) + Von Mises (isotropic) failure indices per trusted row."""

    def __init__(
        self,
        *,
        tertiary: ShellTertiaryResultSet,
        cache: ShellFormulationCache,
        peak_factor: float = 1.5,
    ) -> None:
        self.tertiary = tertiary
        self.cache = cache
        self.peak_factor = peak_factor

    def compute(self) -> ShellQuaternaryResultSet:
        ply_results = self.tertiary.ply_results
        panel_index = self.tertiary.panel_index
        if ply_results is None or ply_results.sig_ply_mat is None or panel_index is None:
            return ShellQuaternaryResultSet(needs_review=np.zeros(0, dtype=bool))

        n_rows, n_ply, _ = ply_results.sig_ply_mat.shape
        trusted_mask = (
            ply_results.trusted_mask
            if ply_results.trusted_mask is not None
            else np.ones(n_rows, dtype=bool)
        )
        hashin_fi = np.zeros((n_rows, max(n_ply, 1)), dtype=float)
        von_mises = np.full(n_rows, np.nan, dtype=float)
        needs_review = ~np.asarray(trusted_mask, dtype=bool)

        panel_laminates = self.cache.panel_laminates
        by_panel: dict[int, list[int]] = {}
        for i, pi in enumerate(panel_index):
            by_panel.setdefault(int(pi), []).append(i)

        interlaminar_fi = np.zeros(n_rows, dtype=float)
        for pi, idxs in by_panel.items():
            idx_arr = np.asarray(idxs, dtype=int)
            kind = _panel_kind(self.cache, pi)

            if kind == "isotropic":
                sigma_allow = _panel_sigma_allow(self.cache, pi)
                if not math.isfinite(sigma_allow) or sigma_allow <= 0.0:
                    needs_review[idx_arr] = True
                    continue
                # Laminate frame ≡ material frame for isotropic; use first ply.
                sig = ply_results.sig_ply_mat[idx_arr, 0, :]
                trusted = trusted_mask[idx_arr]
                fi_vm = von_mises_fi(sig, np.full(len(idx_arr), sigma_allow, dtype=float))
                von_mises[idx_arr] = np.where(trusted, fi_vm, np.nan)
                needs_review[idx_arr] = needs_review[idx_arr] | (~trusted)
                continue

            lam = panel_laminates[pi] if pi < len(panel_laminates) else None
            if lam is None or not getattr(lam, "plies", None):
                needs_review[idxs] = True
                continue
            Xt, Xc, Yt, Yc, S12 = ply_strength_vectors_from_lam(lam)
            fi_p, nr_p = ply_hashin_fi(
                ply_results.sig_ply_mat[idx_arr],
                trusted_mask[idx_arr],
                Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12,
            )
            hashin_fi[idx_arr, : fi_p.shape[1]] = fi_p
            needs_review[idx_arr] = needs_review[idx_arr] | nr_p

            s_il = interlaminar_allowable_from_lam(lam)
            if s_il <= 0.0:
                continue
            shear_trusted = (
                ply_results.shear_trusted_mask[idx_arr]
                if ply_results.shear_trusted_mask is not None
                else np.ones(len(idx_arr), dtype=bool)
            )
            fi_il = np.zeros(len(idx_arr), dtype=float)
            bad = ~shear_trusted
            used_iface = False
            if ply_results.tau_iface is not None and ply_results.tau_iface.size > 0:
                fi_iface, nr_iface, need_fallback = interlaminar_fi_from_interface_tau(
                    ply_results.tau_iface[idx_arr],
                    s_il=s_il,
                    trusted_mask=shear_trusted,
                )
                fi_il = fi_iface
                bad = bad | nr_iface
                used_iface = True
            else:
                need_fallback = shear_trusted.copy()

            if (
                self.tertiary.transverse_shear is not None
                and self.tertiary.thickness_m is not None
                and np.any(need_fallback if used_iface else shear_trusted)
            ):
                qxy = self.tertiary.transverse_shear[idx_arr]
                t_row = self.tertiary.thickness_m[idx_arr]
                fi_scalar, bad_scalar = batched_interlaminar_fi(
                    qxy,
                    t_row,
                    s_il=s_il,
                    peak_factor=self.peak_factor,
                    trusted_mask=shear_trusted,
                )
                if used_iface:
                    fi_il = np.where(need_fallback, fi_scalar, fi_il)
                    bad = np.where(need_fallback, bad_scalar, bad)
                else:
                    fi_il = fi_scalar
                    bad = bad | bad_scalar

            interlaminar_fi[idx_arr] = fi_il
            needs_review[idx_arr] = needs_review[idx_arr] | bad

        governing = _governing_from_arrays(
            hashin_fi,
            interlaminar_fi,
            von_mises,
            panel_index,
            self.tertiary.element_index,
        )
        return ShellQuaternaryResultSet(
            hashin_fi=ShellHashinFiResult(in_plane=hashin_fi, interlaminar=interlaminar_fi),
            von_mises_fi=von_mises,
            governing=governing,
            needs_review=needs_review,
            meta={"n_rows": n_rows, "n_ply": n_ply},
        )


def _governing_from_arrays(
    hashin_fi: np.ndarray,
    interlaminar_fi: np.ndarray,
    von_mises_fi: np.ndarray,
    panel_index: np.ndarray,
    element_index: np.ndarray | None,
) -> dict[str, object]:
    """Governing element = argmax over Hashin in-plane/interlaminar and Von Mises."""
    empty = {
        "fi": 0.0,
        "panel_index": -1,
        "element_index": -1,
        "ply_index": -1,
        "criterion": "",
    }
    best_fi = -1.0
    best: dict[str, object] = dict(empty)

    if hashin_fi.size > 0:
        ri, pi_ply = np.unravel_index(int(np.argmax(hashin_fi)), hashin_fi.shape)
        fv = float(hashin_fi[ri, pi_ply])
        if fv > best_fi:
            best_fi = fv
            best = {
                "fi": fv,
                "panel_index": int(panel_index[ri]),
                "element_index": int(element_index[ri]) if element_index is not None else int(ri),
                "ply_index": int(pi_ply),
                "criterion": "hashin",
            }

    if interlaminar_fi.size > 0:
        ri_il = int(np.argmax(interlaminar_fi))
        fv_il = float(interlaminar_fi[ri_il])
        if fv_il > best_fi:
            best_fi = fv_il
            best = {
                "fi": fv_il,
                "panel_index": int(panel_index[ri_il]),
                "element_index": int(element_index[ri_il]) if element_index is not None else int(ri_il),
                "ply_index": -1,
                "criterion": "hashin",
            }

    vm = np.asarray(von_mises_fi, dtype=float)
    if vm.size > 0 and np.any(np.isfinite(vm)):
        ri_vm = int(np.nanargmax(vm))
        fv_vm = float(vm[ri_vm])
        if math.isfinite(fv_vm) and fv_vm > best_fi:
            best = {
                "fi": fv_vm,
                "panel_index": int(panel_index[ri_vm]),
                "element_index": int(element_index[ri_vm]) if element_index is not None else int(ri_vm),
                "ply_index": 0,
                "criterion": "von_mises",
            }

    if best_fi < 0.0 and best.get("criterion") == "":
        return empty
    return best


__all__ = ["ShellQuaternaryResultsOrchestrator"]
