"""Frozen system-state dataclasses for the lean section-solve pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from ....pre_processing.containers.formulation_objects import (
    AssemblyDiagnostics,
    ShellFormulationCache,
)


@dataclass
class LocalShellSystem:
    cache: ShellFormulationCache


@dataclass
class GlobalShellSystem:
    K: sp.spmatrix
    f: NDArray[np.float64]
    diagnostics: AssemblyDiagnostics | None = None


@dataclass
class ShellDofProvenanceEntry:
    global_dof: int
    fixed: bool = False
    independent: bool = False
    slave: bool = False
    masters: list[tuple[int, float]] = field(default_factory=list)
    reconstructed_value: float = 0.0


@dataclass
class ConstrainedShellSystem:
    constraints: dict[int, list[tuple[int, float]]]
    fixed: set[int]
    dof_provenance: list[ShellDofProvenanceEntry] = field(default_factory=list)


@dataclass
class ReducedShellSystem:
    T: sp.csr_matrix
    K_reduced: sp.csc_matrix
    f_reduced: NDArray[np.float64]
    independent: NDArray[np.int64]


@dataclass
class SolvedReducedShellSystem:
    q: NDArray[np.float64]
    factor: spla.SuperLU | None


@dataclass
class ReconstructedShellField:
    d_global: NDArray[np.float64]
    nan_detected: bool = False
    fixed_dof_leak: bool = False


@dataclass
class PrimaryShellResults:
    reactions: NDArray[np.float64]
    residual: NDArray[np.float64]
    r_full: NDArray[np.float64]


@dataclass
class DisassembledShellResults:
    U_e: list[NDArray[np.float64]]
    element_residual_e: list[NDArray[np.float64]]
    element_residual_norms: list[float]


__all__ = [
    "ConstrainedShellSystem",
    "DisassembledShellResults",
    "GlobalShellSystem",
    "LocalShellSystem",
    "PrimaryShellResults",
    "ReconstructedShellField",
    "ReducedShellSystem",
    "ShellDofProvenanceEntry",
    "SolvedReducedShellSystem",
]
