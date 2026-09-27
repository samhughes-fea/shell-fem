"""Quaternary results orchestrators (tier 4: failure indices, envelopes)."""

from __future__ import annotations

from .bundle_fi import (
    BundleQuaternaryFi,
    compute_quaternary_fi_from_panel_rows,
    flatten_panel_resultant_rows,
    max_interlaminar_fi_from_quaternary,
)
from .compute_governing_element import max_hashin_fi_from_bundle, select_governing_hashin_element
from .compute_hashin_fi import (
    InterfaceShearScreeningResult,
    batched_interlaminar_fi,
    interface_shear_screening_from_tau,
    interface_shear_screening_from_tau_batch,
    interlaminar_fi_from_interface_tau,
    interlaminar_fi_from_row,
    interlaminar_fi_from_row_fsdt,
    ply_hashin_fi,
    ply_hashin_fi_modes,
    ply_strength_vectors_from_lam,
)
from .compute_extreme_fibre_fi import (
    extreme_fibre_fi,
    extreme_fibre_stress_from_ply_mat,
)
from .compute_von_mises_fi import von_mises_fi, von_mises_fi_from_row
from .quaternary_results_orchestrator import ShellQuaternaryResultsOrchestrator

__all__ = [
    "BundleQuaternaryFi",
    "InterfaceShearScreeningResult",
    "ShellQuaternaryResultsOrchestrator",
    "batched_interlaminar_fi",
    "compute_quaternary_fi_from_panel_rows",
    "extreme_fibre_fi",
    "extreme_fibre_stress_from_ply_mat",
    "flatten_panel_resultant_rows",
    "interface_shear_screening_from_tau",
    "interface_shear_screening_from_tau_batch",
    "interlaminar_fi_from_interface_tau",
    "interlaminar_fi_from_row",
    "interlaminar_fi_from_row_fsdt",
    "max_hashin_fi_from_bundle",
    "max_interlaminar_fi_from_quaternary",
    "ply_hashin_fi",
    "ply_hashin_fi_modes",
    "ply_strength_vectors_from_lam",
    "select_governing_hashin_element",
    "von_mises_fi",
    "von_mises_fi_from_row",
]
