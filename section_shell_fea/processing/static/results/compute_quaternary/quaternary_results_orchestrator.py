"""Tier 4 orchestrator: failure indices from tier 3 ply-resolved CLT state.

Pure consumer of ``ShellTertiaryResultSet`` (+ the formulation cache). Mid-face
Hashin / Von Mises (σ_y) plus material-agnostic extreme-fibre (σ_uts / σ_ucs / τ_us)
on laminate free surfaces. See PROCESSING_ARCHITECTURE_PROPOSAL.md tier 4.
"""

from __future__ import annotations

import math

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import ShellFormulationCache
from ..containers.container_hopper import ShellHashinFiResult, ShellQuaternaryResultSet, ShellTertiaryResultSet
from ..containers.ply_results import FACE_INNER, FACE_MID, FACE_OUTER
from .compute_extreme_fibre_fi import (
    extreme_fibre_fi,
    extreme_fibre_stress_from_ply_mat,
    panel_ultimates_finite,
)
from .compute_hashin_fi import (
    batched_interlaminar_fi,
    interface_shear_screening_from_tau,
    interlaminar_allowable_from_lam,
    ply_hashin_fi_modes,
    ply_strength_vectors_from_lam,
)
from .compute_von_mises_fi import von_mises_fi


def _panel_kind(cache: ShellFormulationCache, pi: int) -> str:
    kinds = cache.panel_constitutive_kind
    if pi < len(kinds) and kinds[pi]:
        return str(kinds[pi])
    return "orthotropic"


def _panel_float(seq: list[float], pi: int) -> float:
    if pi < len(seq):
        return float(seq[pi])
    return float("nan")


def _mid_face_sig(sig_ply_mat: np.ndarray) -> np.ndarray:
    """``(n_elem, n_ply, 3)`` mid-face stress from 4D tertiary product."""
    sig = np.asarray(sig_ply_mat, dtype=float)
    if sig.ndim == 4:
        return sig[:, :, FACE_MID, :]
    if sig.ndim == 3:
        return sig
    raise ValueError(f"unexpected sig_ply_mat shape {sig.shape}")


class ShellQuaternaryResultsOrchestrator:
    """Hashin + Von Mises (σ_y) + extreme-fibre (ultimates) failure indices."""

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
        formulation_id = str(self.cache.formulation_id)
        if ply_results is None or ply_results.sig_ply_mat is None or panel_index is None:
            return ShellQuaternaryResultSet(
                needs_review=np.zeros(0, dtype=bool),
                formulation_id=formulation_id,
                meta={"formulation_id": formulation_id},
            )

        sig_full = np.asarray(ply_results.sig_ply_mat, dtype=float)
        sig_mid = _mid_face_sig(sig_full)
        n_rows, n_ply = int(sig_mid.shape[0]), int(sig_mid.shape[1])
        trusted_mask = (
            ply_results.trusted_mask
            if ply_results.trusted_mask is not None
            else np.ones(n_rows, dtype=bool)
        )
        n_cols = max(n_ply, 1)
        hashin_fi = np.zeros((n_rows, n_cols), dtype=float)
        fi_ft = np.zeros((n_rows, n_cols), dtype=float)
        fi_fc = np.zeros((n_rows, n_cols), dtype=float)
        fi_mt = np.zeros((n_rows, n_cols), dtype=float)
        fi_mc = np.zeros((n_rows, n_cols), dtype=float)
        von_mises = np.full(n_rows, np.nan, dtype=float)
        needs_review = ~np.asarray(trusted_mask, dtype=bool)

        sig_extreme = np.zeros((n_rows, 2, 3), dtype=float)
        extreme_fi = np.zeros(n_rows, dtype=float)

        panel_laminates = self.cache.panel_laminates
        by_panel: dict[int, list[int]] = {}
        for i, pi in enumerate(panel_index):
            by_panel.setdefault(int(pi), []).append(i)

        interlaminar_fi = np.zeros(n_rows, dtype=float)
        shear_mode = np.full(n_rows, "unavailable", dtype=object)
        shear_strength_provenance = np.full(n_rows, "unavailable", dtype=object)
        shear_governing_interface = np.full(n_rows, -1, dtype=int)
        shear_governing_side = np.full(n_rows, "", dtype=object)
        shear_component_ratios = np.full((n_rows, 2), np.nan, dtype=float)
        for pi, idxs in by_panel.items():
            idx_arr = np.asarray(idxs, dtype=int)
            kind = _panel_kind(self.cache, pi)
            lam = panel_laminates[pi] if pi < len(panel_laminates) else None
            n_ply_true = len(getattr(lam, "plies", None) or []) or n_ply

            # Extreme-fibre (material-agnostic) — always, every element
            if sig_full.ndim == 4:
                sig_extreme[idx_arr] = extreme_fibre_stress_from_ply_mat(
                    sig_full[idx_arr], n_ply_true=n_ply_true
                )
            uts = _panel_float(self.cache.panel_sigma_uts_pa, pi)
            ucs = _panel_float(self.cache.panel_sigma_ucs_pa, pi)
            tus = _panel_float(self.cache.panel_tau_us_pa, pi)
            if panel_ultimates_finite(uts, ucs, tus):
                fi_ef = extreme_fibre_fi(sig_extreme[idx_arr], uts, ucs, tus)
                trusted = trusted_mask[idx_arr]
                extreme_fi[idx_arr] = np.where(trusted, fi_ef, 0.0)
            else:
                needs_review[idx_arr] = True

            if kind == "isotropic":
                sigma_y = _panel_float(self.cache.panel_sigma_y_pa, pi)
                if not math.isfinite(sigma_y) or sigma_y <= 0.0:
                    needs_review[idx_arr] = True
                    continue
                sig = sig_mid[idx_arr, 0, :]
                trusted = trusted_mask[idx_arr]
                fi_vm = von_mises_fi(sig, np.full(len(idx_arr), sigma_y, dtype=float))
                von_mises[idx_arr] = np.where(trusted, fi_vm, np.nan)
                needs_review[idx_arr] = needs_review[idx_arr] | (~trusted)
                continue

            if lam is None or not getattr(lam, "plies", None):
                needs_review[idxs] = True
                continue
            Xt, Xc, Yt, Yc, S12, S23 = ply_strength_vectors_from_lam(lam)
            if sig_full.ndim == 4:
                modes_outer, nr_o = ply_hashin_fi_modes(
                    sig_full[idx_arr][:, :, FACE_OUTER, :],
                    trusted_mask[idx_arr],
                    Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12, S23=S23,
                )
                modes_inner, nr_i = ply_hashin_fi_modes(
                    sig_full[idx_arr][:, :, FACE_INNER, :],
                    trusted_mask[idx_arr],
                    Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12, S23=S23,
                )
                ft_p = np.maximum(modes_outer[0], modes_inner[0])
                fc_p = np.maximum(modes_outer[1], modes_inner[1])
                mt_p = np.maximum(modes_outer[2], modes_inner[2])
                mc_p = np.maximum(modes_outer[3], modes_inner[3])
                nr_p = nr_o | nr_i
            else:
                modes, nr_p = ply_hashin_fi_modes(
                    sig_mid[idx_arr],
                    trusted_mask[idx_arr],
                    Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12, S23=S23,
                )
                ft_p, fc_p, mt_p, mc_p = modes
            n_mode_cols = ft_p.shape[1]
            fi_ft[idx_arr, :n_mode_cols] = ft_p
            fi_fc[idx_arr, :n_mode_cols] = fc_p
            fi_mt[idx_arr, :n_mode_cols] = mt_p
            fi_mc[idx_arr, :n_mode_cols] = mc_p
            fi_p = np.maximum(np.maximum(ft_p, fc_p), np.maximum(mt_p, mc_p))
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
            authoritative_interfaces = (
                n_ply_true >= 2
                and ply_results.tau_iface is not None
                and ply_results.tau_iface.size > 0
            )
            if authoritative_interfaces:
                for local_index, row_index in enumerate(idx_arr):
                    result = interface_shear_screening_from_tau(
                        ply_results.tau_iface[row_index, : n_ply_true + 1],
                        lam,
                        trusted=bool(shear_trusted[local_index]),
                    )
                    if np.isfinite(result.failure_index):
                        fi_il[local_index] = result.failure_index
                    bad[local_index] = result.needs_review
                    shear_mode[row_index] = result.recovery_mode
                    shear_strength_provenance[row_index] = result.strength_provenance
                    shear_governing_interface[row_index] = (
                        -1
                        if result.governing_interface is None
                        else result.governing_interface
                    )
                    shear_governing_side[row_index] = result.governing_side or ""
                    shear_component_ratios[row_index] = (
                        result.tau13_ratio,
                        result.tau23_ratio,
                    )
            elif (
                self.tertiary.transverse_shear is not None
                and self.tertiary.thickness_m is not None
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
                fi_il = fi_scalar
                bad = bad | bad_scalar
                shear_mode[idx_arr] = "compatibility_scalar"
                shear_strength_provenance[idx_arr] = "compatibility_s13_scalar"

            interlaminar_fi[idx_arr] = fi_il
            needs_review[idx_arr] = needs_review[idx_arr] | bad

        governing = _governing_from_arrays(
            hashin_fi,
            von_mises,
            extreme_fi,
            panel_index,
            self.tertiary.element_index,
            hashin_modes=(fi_ft, fi_fc, fi_mt, fi_mc),
        )
        return ShellQuaternaryResultSet(
            hashin_fi=ShellHashinFiResult(
                in_plane=hashin_fi,
                interlaminar=interlaminar_fi,
                interface_shear=interlaminar_fi.copy(),
                interface_shear_recovery_mode=shear_mode,
                interface_shear_strength_provenance=shear_strength_provenance,
                interface_shear_governing_interface=shear_governing_interface,
                interface_shear_governing_side=shear_governing_side,
                interface_shear_component_ratios=shear_component_ratios,
                fi_ft=fi_ft,
                fi_fc=fi_fc,
                fi_mt=fi_mt,
                fi_mc=fi_mc,
            ),
            von_mises_fi=von_mises,
            sig_extreme_fibre=sig_extreme,
            extreme_fibre_fi=extreme_fi,
            governing=governing,
            needs_review=needs_review,
            meta={"n_rows": n_rows, "n_ply": n_ply, "formulation_id": formulation_id},
            max_utilization_governing=(
                None
                if not isinstance(governing.get("utilization"), (int, float))
                else float(governing["utilization"])
            ),
            formulation_id=formulation_id,
        )


def _governing_from_arrays(
    hashin_fi: np.ndarray,
    von_mises_fi: np.ndarray,
    extreme_fibre_fi: np.ndarray,
    panel_index: np.ndarray,
    element_index: np.ndarray | None,
    hashin_modes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, object]:
    """Governing element uses utilization among required available criteria.

    Interface-shear diagnostics never enter this envelope. Extreme-fibre is
    opt-in and is recorded but does not override plane-stress Hashin / VM.
    """
    empty = {
        "fi": 0.0,
        "utilization": float("nan"),
        "status": "unavailable",
        "panel_index": -1,
        "element_index": -1,
        "ply_index": -1,
        "face": "",
        "mode": "",
        "criterion": "",
    }
    candidates: list[tuple[float, dict[str, object]]] = []

    if hashin_fi.size > 0 and np.any(np.isfinite(hashin_fi)):
        ri, pi_ply = np.unravel_index(int(np.nanargmax(np.where(np.isfinite(hashin_fi), hashin_fi, -np.inf))), hashin_fi.shape)
        fv = float(hashin_fi[ri, pi_ply])
        if math.isfinite(fv):
            rf = float("inf") if fv <= 0.0 else 1.0 / math.sqrt(fv) if fv > 0.0 else float("inf")
            # Mixed MC is not purely quadratic; utilization from sqrt is conservative
            # for FT/FC/MT. Use 1/RF with RF = 1/sqrt(FI) for homogeneous modes.
            mode = "envelope"
            if hashin_modes is not None:
                names = ("FT", "FC", "MT", "MC")
                mode = names[int(np.argmax([m[ri, pi_ply] for m in hashin_modes]))]
            util = 0.0 if not math.isfinite(rf) else (0.0 if rf == float("inf") else 1.0 / rf)
            if fv > 0.0:
                util = math.sqrt(fv)
            candidates.append(
                (
                    util,
                    {
                        "fi": fv,
                        "utilization": util,
                        "status": "available",
                        "panel_index": int(panel_index[ri]),
                        "element_index": int(element_index[ri]) if element_index is not None else int(ri),
                        "ply_index": int(pi_ply),
                        "face": "outer_inner",
                        "mode": mode,
                        "criterion": "hashin",
                        "criterion_id": "hashin_1980_plane_stress",
                    },
                )
            )

    vm = np.asarray(von_mises_fi, dtype=float)
    if vm.size > 0 and np.any(np.isfinite(vm)):
        ri_vm = int(np.nanargmax(vm))
        fv_vm = float(vm[ri_vm])
        if math.isfinite(fv_vm):
            candidates.append(
                (
                    fv_vm,
                    {
                        "fi": fv_vm,
                        "utilization": fv_vm,
                        "status": "available",
                        "panel_index": int(panel_index[ri_vm]),
                        "element_index": int(element_index[ri_vm]) if element_index is not None else int(ri_vm),
                        "ply_index": 0,
                        "face": "mid",
                        "mode": "von_mises",
                        "criterion": "von_mises",
                    },
                )
            )

    if not candidates:
        return empty
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = dict(candidates[0][1])
    best["fi"] = best["utilization"] if best["criterion"] == "von_mises" else best["fi"]
    return best


__all__ = ["ShellQuaternaryResultsOrchestrator"]
