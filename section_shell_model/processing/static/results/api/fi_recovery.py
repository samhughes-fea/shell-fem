"""Failure-index helpers: provenance gating, Hashin from shell resultants, path comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from blade_structure_utilities.section_shell_model.pre_processing.containers.types import FieldProvenance, ShellPanelResultants, provenance_is_trusted
from ..compute_tertiary.local_clpt_shell import default_skin_strengths_pa, solve_station_clpt_shell

# Provenance gating moved to tier 3 (see PROCESSING_ARCHITECTURE_PROPOSAL.md); re-exported
# here for compatibility with existing callers (docs/architecture/DEPRECATION_REGISTER.md).
from ..compute_tertiary.provenance import (
    clpt_inputs_trusted,
    interlaminar_inputs_trusted,
    resultant_field_trusted,
)
# Failure-criterion evaluation moved to tier 4; re-exported here for compatibility.
from ..compute_quaternary.compute_governing_element import (
    max_hashin_fi_from_bundle,
    select_governing_hashin_element,
)
from ..compute_quaternary.bundle_fi import (
    BundleQuaternaryFi,
    compute_quaternary_fi_from_panel_rows,
    max_interlaminar_fi_from_quaternary,
)
from ..compute_quaternary.compute_hashin_fi import (
    interlaminar_allowable_from_lam,
    interlaminar_allowable_pa,
    interlaminar_fi_from_interface_tau,
    interlaminar_fi_from_row,
    interlaminar_fi_from_row_fsdt,
    ply_strength_vectors_from_lam as _ply_strength_vectors_from_lam,
)
from ..compute_quaternary.compute_von_mises_fi import (
    von_mises_fi_from_row,
)


def hashin_fi_per_ply_from_rows_batch(
    rows: list[ShellPanelResultants | None],
    lam: Any,
    plies: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched per-ply Hashin for many element rows sharing one laminate.

    The CLT solve (ABD^-1 [N;M] -> eps0/kappa) delegates to tier 3's
    ``batched_clt_strains`` (same batching/ordering); ply stress recovery and the
    Hashin evaluation below are unchanged from before the tier split, preserving
    ``fi_out``/``needs_review`` bit-for-bit (see PROCESSING_ARCHITECTURE_PROPOSAL.md
    non-goals: no numerics changes).
    """
    from blade_structure_utilities.section_materials.failure_criteria import hashin_fi
    from blade_structure_utilities.section_materials.laminate import LaminateDefinition
    from blade_structure_utilities.section_materials.clpt_recovery import ply_mid_stresses_laminate

    from ..compute_tertiary.compute_clt_strains import batched_clt_strains

    n_rows = len(rows)
    n_ply = len(lam.plies) if isinstance(lam, LaminateDefinition) else 0
    fi_out = np.zeros((n_rows, max(n_ply, 1)), dtype=float)
    needs_review = np.zeros(n_rows, dtype=bool)
    if not isinstance(lam, LaminateDefinition) or not lam.plies:
        needs_review[:] = True
        return fi_out, needs_review

    eps0_all, kappa_all, trusted_mask = batched_clt_strains(rows, lam)
    needs_review = ~trusted_mask
    if not np.any(trusted_mask):
        return fi_out, needs_review

    Xt, Xc, Yt, Yc, S12 = _ply_strength_vectors_from_lam(lam)
    for ri in np.flatnonzero(trusted_mask):
        eps0 = eps0_all[ri]
        kappa = kappa_all[ri]
        sig_lam = ply_mid_stresses_laminate(lam, eps0, kappa)
        for pi, sig in enumerate(sig_lam):
            if pi >= n_ply:
                break
            sig3 = np.asarray(sig, dtype=float).ravel()[:3]
            fi_out[ri, pi] = float(
                hashin_fi(
                    sig3.reshape(1, 3),
                    Xt[pi : pi + 1],
                    Xc[pi : pi + 1],
                    Yt[pi : pi + 1],
                    Yc[pi : pi + 1],
                    S12[pi : pi + 1],
                )[0]
            )
    return fi_out, needs_review


def hashin_fi_per_ply_from_row(
    row: ShellPanelResultants,
    lam: Any,
    plies: list,
) -> tuple[np.ndarray, bool]:
    """Per-ply Hashin envelope via CLPT recovery + ``materials.hashin_fi``."""
    from blade_structure_utilities.section_materials.failure_criteria import hashin_fi
    from blade_structure_utilities.section_materials.laminate import LaminateDefinition

    if not isinstance(lam, LaminateDefinition) or not lam.plies:
        return np.zeros(0, dtype=float), True
    if not clpt_inputs_trusted(row):
        return np.zeros(len(lam.plies), dtype=float), True
    Xt, Xc, Yt, Yc, S12 = _ply_strength_vectors_from_lam(lam)
    res = solve_station_clpt_shell(
        row, plies,
        Xt=float(Xt[0]), Xc=float(Xc[0]), Yt=float(Yt[0]),
        Yc=float(Yc[0]), S12=float(S12[0]),
        lam=lam,
    )
    fi_out = np.zeros(len(lam.plies), dtype=float)
    for i, sig in enumerate(res.sig_lam_mid):
        if i >= len(lam.plies):
            break
        sig3 = np.asarray(sig, dtype=float).ravel()[:3]
        fi_out[i] = float(
            hashin_fi(
                sig3.reshape(1, 3),
                Xt[i : i + 1], Xc[i : i + 1], Yt[i : i + 1], Yc[i : i + 1], S12[i : i + 1],
            )[0]
        )
    return fi_out, False


def hashin_fi_from_row(
    row: ShellPanelResultants,
    plies: list,
    *,
    Xt: float,
    Xc: float,
    Yt: float,
    Yc: float,
    S12: float,
    lam: Any | None = None,
) -> tuple[float, bool]:
    """Return ``(max_fi, needs_review)`` for one element row.

    When CLPT inputs are not provenance-trusted, returns ``(0.0, True)`` so the
    value cannot govern but the caller can surface a review flag.
    """
    if not clpt_inputs_trusted(row):
        return 0.0, True
    from blade_structure_utilities.section_materials.laminate import LaminateDefinition, laminate_definition_from_legacy

    ld = lam if isinstance(lam, LaminateDefinition) else laminate_definition_from_legacy(lam, plies=plies)
    if ld is None:
        return 0.0, True
    res = solve_station_clpt_shell(
        row, plies, Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S12=S12, lam=ld,
    )
    if len(res.fi) == 0:
        return 0.0, False
    fi_max = float(np.max(res.fi))
    if not np.isfinite(fi_max):
        return 0.0, True
    return fi_max, False


# Path-compare helpers moved to processing/static/diagnostics/path_compare.py (R15).
from ...diagnostics.path_compare import (  # noqa: F401
    PathCompareStats,
    compare_path_resultants,
    section_integrals_from_bundle,
)


__all__ = [
    "BundleQuaternaryFi",
    "PathCompareStats",
    "clpt_inputs_trusted",
    "compare_path_resultants",
    "compute_quaternary_fi_from_panel_rows",
    "hashin_fi_from_row",
    "hashin_fi_per_ply_from_row",
    "interlaminar_allowable_from_lam",
    "interlaminar_allowable_pa",
    "interlaminar_fi_from_interface_tau",
    "interlaminar_fi_from_row",
    "interlaminar_fi_from_row_fsdt",
    "interlaminar_inputs_trusted",
    "max_hashin_fi_from_bundle",
    "max_interlaminar_fi_from_quaternary",
    "resultant_field_trusted",
    "section_integrals_from_bundle",
    "select_governing_hashin_element",
    "von_mises_fi_from_row",
]
