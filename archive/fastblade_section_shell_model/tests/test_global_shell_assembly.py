"""Direct regression on the single global shell assembly (``engine/global_shell_assembly.py``).

This 1,000+-line assembly spine had no direct test; here we lock its structural
contract (K_global shape/symmetry, load-vector shape, per-panel maps) and the MPC
interface-constraint reduction on a real multi-cell section.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section
from blade_precompute.section_shell_model.pre_processing.containers.fem_config import MITC4_DEFAULTS
from blade_precompute.section_shell_model.engine.global_shell_assembly import (
    assemble_global_shell,
)

_NDOF = MITC4_DEFAULTS.n_dof_node


def _panels():
    pb = build_pipeline_b_section("2412", [0.35], chord_m=1.0)
    panels = pb.panels
    zeros = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels]
    return panels, zeros


def test_assemble_global_shell_structural_contract() -> None:
    panels, zeros = _panels()
    K, f, node_meta, panel_maps, abd_p, t_p, g_p, lab_p = assemble_global_shell(
        panels, zeros, zeros, n_elements_per_panel=4,
    )

    ndof = node_meta.n_nodes * _NDOF
    assert sp.issparse(K)
    assert K.shape == (ndof, ndof)
    assert np.asarray(f).shape == (ndof,)
    assert len(panel_maps) == len(panels)
    assert abd_p.ndim == 3 and abd_p.shape[1:] == (6, 6)
    assert len(lab_p) == abd_p.shape[0]
    assert np.all(np.isfinite(K.data))
    assert np.all(np.isfinite(np.asarray(f)))


def test_assemble_global_shell_is_symmetric() -> None:
    panels, zeros = _panels()
    K, _f, _nm, _pm, *_abd = assemble_global_shell(panels, zeros, zeros, n_elements_per_panel=4)

    K = K.tocsr()
    asym = (K - K.T)
    scale = max(float(abs(K).max()), 1.0)
    assert float(abs(asym).max()) / scale < 1e-9


def test_assemble_global_shell_element_count_scales_dofs() -> None:
    """More elements per panel ⇒ strictly more global DOFs (mesh refinement wiring)."""
    panels, zeros = _panels()
    _K4, _f4, nm4, _, *_abd4 = assemble_global_shell(panels, zeros, zeros, n_elements_per_panel=4)
    _K8, _f8, nm8, _, *_abd8 = assemble_global_shell(panels, zeros, zeros, n_elements_per_panel=8)

    assert nm8.n_nodes > nm4.n_nodes


def test_assemble_global_shell_interface_modes_change_junction_topology() -> None:
    """``shared`` merges junction nodes; ``transformed_basis`` keeps them separate + constrained."""
    panels, zeros = _panels()
    _Ks, _fs, nm_shared, _, *_abd = assemble_global_shell(
        panels, zeros, zeros, n_elements_per_panel=4, interface_constraint_mode="shared",
    )
    _Kt, _ft, nm_tb, _, *_abd2 = assemble_global_shell(
        panels, zeros, zeros, n_elements_per_panel=4,
        interface_constraint_mode="transformed_basis",
    )
    # Shared physically collapses coincident junction nodes into one, so it has
    # strictly fewer global nodes than the constraint-based transformed basis.
    assert nm_shared.n_nodes < nm_tb.n_nodes


def test_assembly_only_fast_path_is_deterministic() -> None:
    """Assembly-only path is the sole K_global source for periodic homogenisation.

    Locks perf finding #1: ``assemble_global_shell`` (via ``assemble_section_from_panels``)
    is idempotent so cache hits / double assembly agree to numerical noise.
    """
    panels, zeros = _panels()
    K_a, _f_a, nm_a, _pm_a, abd_a, *_rest_a = assemble_global_shell(
        panels, zeros, zeros, n_elements_per_panel=4,
    )
    K_b, _f_b, nm_b, _pm_b, abd_b, *_rest_b = assemble_global_shell(
        panels, zeros, zeros, n_elements_per_panel=4,
    )

    assert nm_a.n_nodes == nm_b.n_nodes
    assert K_a.shape == K_b.shape
    assert abd_a.shape == abd_b.shape
    diff = (K_a.tocsr() - K_b.tocsr())
    scale = max(float(abs(K_a).max()), 1.0)
    assert float(abs(diff).max()) / scale < 1e-12
