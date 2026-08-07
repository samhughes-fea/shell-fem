"""Ply-resolution CLT recovery payload (tier 3 native output).

Adds a "ply" resolution alongside global / elemental / nodal / gaussian: the
material-frame ply strain/stress state that tier 3 (``compute_tertiary``)
produces from the CLT solve, and that tier 4 (``compute_quaternary``) consumes
to evaluate failure criteria. See PROCESSING_ARCHITECTURE_PROPOSAL.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ShellPlyResults:
    """Material-frame ply strain/stress state recovered per trusted element row."""

    eps0: np.ndarray | None = None
    """(n_elem, 3) laminate mid-surface strains from the batched ABD solve."""

    kappa: np.ndarray | None = None
    """(n_elem, 3) laminate curvatures from the batched ABD solve."""

    eps_ply_mat: np.ndarray | None = None
    """(n_elem, n_ply, 3) per-ply strains, rotated to ply material axes."""

    sig_ply_mat: np.ndarray | None = None
    """(n_elem, n_ply, 3) per-ply stresses, rotated to ply material axes."""

    trusted_mask: np.ndarray | None = None
    """(n_elem,) bool — provenance-trusted rows that entered the CLT solve."""

    tau_iface: np.ndarray | None = None
    """(n_elem, n_iface, 2) interface ``[tau_13, tau_23]`` [Pa]; ``n_iface = n_ply + 1``.

    Free surfaces are zero; interior values come from FSDT parabolic recovery.
    Untrusted interlaminar rows remain zero with ``shear_trusted_mask`` false.
    """

    shear_trusted_mask: np.ndarray | None = None
    """(n_elem,) bool — rows whose Qx/Qy passed interlaminar provenance gating."""


__all__ = ["ShellPlyResults"]
