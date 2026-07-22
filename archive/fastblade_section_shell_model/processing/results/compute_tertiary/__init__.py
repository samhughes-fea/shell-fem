"""Tertiary results orchestrators (tier 3: CLT recovery at the material frame)."""

from __future__ import annotations

from .compute_clt_strains import batched_clt_strains
from .compute_ply_stresses import ply_strains_stresses_material_frame
from .provenance import clpt_inputs_trusted, interlaminar_inputs_trusted, resultant_field_trusted
from .recovery_operators import (
    CcxIpShearRecoveryOperator,
    FSDTInterfaceShearRecoveryOperator,
    HSDTQxShearRecoveryOperator,
    InPlaneLaminateRecoveryOperator,
    SHEAR_RECOVERY_MODE_CCX_IP,
    SHEAR_RECOVERY_MODE_FSDT_PARABOLIC,
    SHEAR_RECOVERY_MODE_HSDT_QX,
    TransverseShearRecoveryProtocol,
    fsdt_interface_shear_profile,
    resolve_shear_recovery_operator,
)
from .tertiary_results_orchestrator import ShellTertiaryResultsOrchestrator

__all__ = [
    "CcxIpShearRecoveryOperator",
    "FSDTInterfaceShearRecoveryOperator",
    "HSDTQxShearRecoveryOperator",
    "InPlaneLaminateRecoveryOperator",
    "SHEAR_RECOVERY_MODE_CCX_IP",
    "SHEAR_RECOVERY_MODE_FSDT_PARABOLIC",
    "SHEAR_RECOVERY_MODE_HSDT_QX",
    "ShellTertiaryResultsOrchestrator",
    "TransverseShearRecoveryProtocol",
    "batched_clt_strains",
    "clpt_inputs_trusted",
    "fsdt_interface_shear_profile",
    "interlaminar_inputs_trusted",
    "ply_strains_stresses_material_frame",
    "resolve_shear_recovery_operator",
    "resultant_field_trusted",
]
