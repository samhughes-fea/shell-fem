"""Tertiary results orchestrators (tier 3: CLT recovery).

Mid-surface ``eps0``/``kappa`` and ``tau_iface`` are in **E**; face
``eps_ply_mat``/``sig_ply_mat`` are in **M**. See ``coordinate_frames``.
"""

from __future__ import annotations

from .compute_clt_strains import batched_clt_strains
from .compute_ply_stresses import ply_strains_stresses_material_frame
from .local_clpt_shell import StationCLPTShellResult, default_skin_strengths_pa, solve_station_clpt_shell
from .provenance import clpt_inputs_trusted, interlaminar_inputs_trusted, resultant_field_trusted
from .recovery_operators import (
    CcxIpShearRecoveryOperator,
    FSDTInterfaceShearRecoveryOperator,
    HSDTQ1ShearRecoveryOperator,
    InPlaneLaminateRecoveryOperator,
    SHEAR_RECOVERY_MODE_CCX_IP,
    SHEAR_RECOVERY_MODE_FSDT_PARABOLIC,
    SHEAR_RECOVERY_MODE_HSDT_Q1,
    TransverseShearRecoveryProtocol,
    fsdt_interface_shear_profile,
    resolve_shear_recovery_operator,
)
from .tertiary_results_orchestrator import ShellTertiaryResultsOrchestrator

__all__ = [
    "CcxIpShearRecoveryOperator",
    "FSDTInterfaceShearRecoveryOperator",
    "HSDTQ1ShearRecoveryOperator",
    "InPlaneLaminateRecoveryOperator",
    "SHEAR_RECOVERY_MODE_CCX_IP",
    "SHEAR_RECOVERY_MODE_FSDT_PARABOLIC",
    "SHEAR_RECOVERY_MODE_HSDT_Q1",
    "ShellTertiaryResultsOrchestrator",
    "StationCLPTShellResult",
    "TransverseShearRecoveryProtocol",
    "batched_clt_strains",
    "clpt_inputs_trusted",
    "default_skin_strengths_pa",
    "fsdt_interface_shear_profile",
    "interlaminar_inputs_trusted",
    "ply_strains_stresses_material_frame",
    "resolve_shear_recovery_operator",
    "resultant_field_trusted",
    "solve_station_clpt_shell",
]
