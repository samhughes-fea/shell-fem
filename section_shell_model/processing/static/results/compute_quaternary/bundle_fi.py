"""Channel-A bridge: ``SectionShellRecoveryBundle`` rows → four-tier FI.

Flattens per-panel resultant rows into the secondary → tertiary → quaternary
container pipeline. ``compute_quaternary_fi_from_panel_rows`` is a **legacy**
Channel-A entry (traction / parity); production FI uses the linear-static
runner quaternary set. See RESULTS_DESIGN.md and DEPRECATED_ALIASES.md.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.section_shell_model.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)
from blade_structure_utilities.section_shell_model.pre_processing.containers.types import ShellPanelResultants
from ..compute_tertiary.recovery_operators import (
    FSDTInterfaceShearRecoveryOperator,
    TransverseShearRecoveryProtocol,
)
from ..compute_tertiary.tertiary_results_orchestrator import ShellTertiaryResultsOrchestrator
from ..containers.container_hopper import ShellQuaternaryResultSet, ShellSecondaryResultSet
from ..containers.elemental_results import ShellElementalResults
from .quaternary_results_orchestrator import ShellQuaternaryResultsOrchestrator


@dataclass(frozen=True)
class BundleQuaternaryFi:
    """Flattened quaternary FI plus the panel-row index map used to build it."""

    quaternary: ShellQuaternaryResultSet
    row_panel_index: NDArray[np.int64]
    n_rows: int
    clpt_trusted_mask: NDArray[np.bool_]
    shear_trusted_mask: NDArray[np.bool_]


def _normalise_row(row: ShellPanelResultants, panel_index: int) -> ShellPanelResultants:
    """Ensure ``panel_index`` is set for tier-3 panel laminate lookup."""
    if int(getattr(row, "panel_index", -1)) == int(panel_index):
        return row
    from dataclasses import replace

    return replace(row, panel_index=int(panel_index))


def flatten_panel_resultant_rows(
    all_panel_results: Sequence[Sequence[ShellPanelResultants | None] | None],
) -> tuple[list[ShellPanelResultants], NDArray[np.int64]]:
    """Flatten nested panel rows; rewrite ``panel_index`` to the outer panel loop index."""
    rows: list[ShellPanelResultants] = []
    panel_idx: list[int] = []
    for pi, panel_res in enumerate(all_panel_results or []):
        for row in panel_res or []:
            if row is None:
                continue
            rows.append(_normalise_row(row, pi))
            panel_idx.append(pi)
    return rows, np.asarray(panel_idx, dtype=np.int64)


def compute_quaternary_fi_from_panel_rows(
    all_panel_results: Sequence[Sequence[ShellPanelResultants | None] | None],
    panel_laminates: Sequence[Any],
    *,
    panel_constitutive_kind: Sequence[str] | None = None,
    panel_sigma_allow_pa: Sequence[float] | None = None,
    peak_factor: float = 1.5,
    shear_operator: TransverseShearRecoveryProtocol | None = None,
) -> BundleQuaternaryFi:
    """Legacy Channel-A FI on flattened bundle ``all_panel_results``.

    Prefer runner ``quaternary_results`` via the linear-static cascade.
    """
    warnings.warn(
        "compute_quaternary_fi_from_panel_rows is a deprecated Channel-A leftover; "
        "prefer SectionLinearStaticRunner quaternary_results on the linear-static "
        "track. See docs/DEPRECATED_ALIASES.md.",
        DeprecationWarning,
        stacklevel=2,
    )
    rows, row_panel_index = flatten_panel_resultant_rows(all_panel_results)
    n_panels = len(panel_laminates)
    kinds = list(panel_constitutive_kind) if panel_constitutive_kind is not None else (
        ["orthotropic"] * n_panels
    )
    allows = list(panel_sigma_allow_pa) if panel_sigma_allow_pa is not None else (
        [float("nan")] * n_panels
    )
    # Pad constitutive metadata to panel_laminates length.
    while len(kinds) < n_panels:
        kinds.append("orthotropic")
    while len(allows) < n_panels:
        allows.append(float("nan"))

    cache = ShellFormulationCache(
        stiffness_objects=[],
        force_objects=[],
        panel_laminates=list(panel_laminates),
        panel_constitutive_kind=list(kinds),  # type: ignore[arg-type]
        panel_sigma_allow_pa=list(allows),
    )
    secondary = ShellSecondaryResultSet(
        elemental_results=ShellElementalResults(resultant_rows=rows),
    )
    tertiary = ShellTertiaryResultsOrchestrator(
        secondary=secondary,
        cache=cache,
        shear_operator=shear_operator or FSDTInterfaceShearRecoveryOperator(),
    ).compute()
    quaternary = ShellQuaternaryResultsOrchestrator(
        tertiary=tertiary,
        cache=cache,
        peak_factor=peak_factor,
    ).compute()
    ply = tertiary.ply_results
    clpt_trusted = (
        np.asarray(ply.trusted_mask, dtype=bool)
        if ply is not None and ply.trusted_mask is not None
        else np.zeros(len(rows), dtype=bool)
    )
    shear_trusted = (
        np.asarray(ply.shear_trusted_mask, dtype=bool)
        if ply is not None and ply.shear_trusted_mask is not None
        else np.zeros(len(rows), dtype=bool)
    )
    return BundleQuaternaryFi(
        quaternary=quaternary,
        row_panel_index=row_panel_index,
        n_rows=len(rows),
        clpt_trusted_mask=clpt_trusted,
        shear_trusted_mask=shear_trusted,
    )


def max_interlaminar_fi_from_quaternary(bundle_fi: BundleQuaternaryFi) -> float:
    """Peak interlaminar FI across all trusted quaternary rows."""
    q = bundle_fi.quaternary
    if q.hashin_fi is None or q.hashin_fi.interlaminar is None:
        return 0.0
    arr = np.asarray(q.hashin_fi.interlaminar, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmax(arr))


__all__ = [
    "BundleQuaternaryFi",
    "compute_quaternary_fi_from_panel_rows",
    "flatten_panel_resultant_rows",
    "max_interlaminar_fi_from_quaternary",
]
