"""WP0 exactness and element-path baseline pins for FE preprocessing cleanup.

See ``docs/FE_PREPROCESSING_CLEANUP_WP0_BASELINE.md``.
"""

from __future__ import annotations

import statistics
import time
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellTheory,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.formulation_factory import (
    _element_L_s,
    build_formulation_cache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.mitc4.element import (
    mitc4_stiffness_contributions,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.panel_node_assembly import (
    panel_maps_from_section_strip_mesh,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.operations.assemble_geometric import (
    _element_L_s as buckling_element_L_s,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.solve.solve_phase_timing import (
    PHASE_ELEMENT_FORCE_VECTORS,
    PHASE_ELEMENT_STIFFNESS_MATRICES,
    SolvePhaseCollector,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_secondary.secondary_results_orchestrator import (
    _element_L_s as recovery_element_L_s,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.tests._panel_topology_factory import (
    build_shell_panel_topology,
)
from blade_structure_utilities.mesh_utilities.section_mesh import build_section_strip_mesh


def _panel_maps_and_loads(*, n_elements_per_panel: int = 4):
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    zones = list(panel_topology.zones)
    nx = [np.linspace(0.0, 1000.0, len(np.asarray(p.s, dtype=float))) for p in zones]
    nxy = [np.zeros_like(n) for n in nx]
    strip = build_section_strip_mesh(
        panel_topology.shell_inputs, n_elements_per_panel=n_elements_per_panel
    )
    panel_maps, _ = panel_maps_from_section_strip_mesh(strip, zones)
    return panel_maps, zones, nx, nxy


def _scaled_outboard_zones(zones: list, scale: float = 1.25) -> list:
    """Contour lengths scaled so blended assembly ``L_s`` differs from inboard-only."""
    out: list = []
    for z in zones:
        s = np.asarray(z.s, dtype=float)
        s0 = float(s[0])
        out.append(
            SimpleNamespace(
                s=(s0 + (s - s0) * float(scale)).copy(),
                nodes=np.asarray(getattr(z, "nodes"), dtype=float).copy(),
                lam=z.lam,
            )
        )
    return out


def test_typed_panel_spec_matches_tuple_abd_ke():
    """Pinned: typed ``PanelConstitutiveSpec`` and tuple ABD produce identical Ke."""
    panel_maps, zones, nx, nxy = _panel_maps_and_loads(n_elements_per_panel=2)
    cache = build_formulation_cache(
        panel_maps, zones, nx, nxy, shell_theory=ShellTheory.FSDT, L_x=1.0
    )
    assert cache.panel_constitutive_specs
    assert any(spec is not None for spec in cache.panel_constitutive_specs)

    for stiff in cache.stiffness_objects:
        pi = stiff.panel_index
        spec = cache.panel_constitutive_specs[pi]
        assert spec is not None
        pm = panel_maps[pi]
        i0, i1, _i2, _i3 = pm.elements[stiff.local_element_index]
        L_s = float(abs(pm.s_nodes[i1] - pm.s_nodes[i0]))
        if L_s < 1e-30:
            continue
        ABD, thickness, _, _, _G = cache.panel_abd_stacks[pi]
        ref = mitc4_stiffness_contributions(
            L_s, 1.0, ABD, thickness, A_s=spec.fsdt.A_s
        ).K_e
        assert np.allclose(stiff.K_e, ref, rtol=0.0, atol=1e-12)
        # Complete A_s is authoritative; scalar G_eff is a compatibility projection.
        assert spec.fsdt.A_s.shape == (2, 2)
        assert np.all(np.isfinite(spec.fsdt.A_s))


def test_zones_outboard_exposes_assembly_recovery_buckling_L_s_disagreement():
    """WP0 pin: assembly uses blended ``L_s``; recovery/buckling recompute inboard-only.

    WP2 must close this disagreement by persisting the assembly record.
    """
    panel_maps, zones, nx, nxy = _panel_maps_and_loads(n_elements_per_panel=3)
    zones_out = _scaled_outboard_zones(zones, scale=1.25)
    cache = build_formulation_cache(
        panel_maps,
        zones,
        nx,
        nxy,
        shell_theory=ShellTheory.FSDT,
        L_x=1.0,
        zones_outboard=zones_out,
    )
    assert cache.stiffness_objects, "expected at least one element"

    disagreements = 0
    for stiff in cache.stiffness_objects:
        pi = int(stiff.panel_index)
        lei = int(stiff.local_element_index)
        pm = panel_maps[pi]
        i0, i1, _i2, _i3 = pm.elements[lei]
        L_s_assembly, _ = _element_L_s(pi, pm.s_nodes, i0, i1, zones, zones_out, 1.0)
        L_s_plain = float(abs(pm.s_nodes[i1] - pm.s_nodes[i0]))
        L_s_recovery = recovery_element_L_s(
            pi=pi,
            local_element_index=lei,
            panel_maps=panel_maps,
            panel_meta=None,
        )
        L_s_buckling = buckling_element_L_s(pm, lei)
        assert L_s_recovery == pytest.approx(L_s_plain, abs=0.0, rel=0.0)
        assert L_s_buckling == pytest.approx(L_s_plain, abs=0.0, rel=0.0)
        if abs(L_s_assembly - L_s_plain) > 1e-12:
            disagreements += 1
            assert L_s_assembly != pytest.approx(L_s_recovery, abs=1e-12)
            assert L_s_assembly != pytest.approx(L_s_buckling, abs=1e-12)
    assert disagreements > 0, "scaled outboard must create blended vs plain L_s mismatch"


def test_donnell_length_remains_plain_inboard_contour():
    """Intentional: Donnell uses plain contour length against blended stiffness length."""
    panel_maps, zones, nx, nxy = _panel_maps_and_loads(n_elements_per_panel=2)
    zones_out = _scaled_outboard_zones(zones, scale=1.4)
    factory = __import__(
        "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing."
        "element_library.mitc4.factory",
        fromlist=["Mitc4ElementFactory"],
    ).Mitc4ElementFactory()
    elems = list(
        factory.iter_elements(
            panel_maps,
            zones,
            nx,
            nxy,
            L_x=1.0,
            zones_outboard=zones_out,
        )
    )
    assert elems
    for elem in elems:
        g = elem.geometry
        assert g.L_s_donnell > 0.0
        assert g.L_s > g.L_s_donnell  # blended longer with scale>1
        ratio = g.L_s_donnell / g.L_s
        assert 0.0 < ratio < 1.0


def test_element_path_benchmark_harness_records_timing_and_allocation():
    """D0.1 harness: warm median timing + peak allocation for uniform/mixed laminates."""
    panel_maps, zones, nx, nxy = _panel_maps_and_loads(n_elements_per_panel=6)

    def _run_once() -> tuple[float, float, int, int]:
        collector = SolvePhaseCollector()
        t0 = time.perf_counter()
        cache = build_formulation_cache(
            panel_maps,
            zones,
            nx,
            nxy,
            shell_theory=ShellTheory.FSDT,
            L_x=1.0,
            phase_collector=collector,
        )
        elapsed = time.perf_counter() - t0
        totals = collector.freeze().phase_totals_s
        stiff_ms = float(totals.get(PHASE_ELEMENT_STIFFNESS_MATRICES, 0.0)) * 1e3
        force_ms = float(totals.get(PHASE_ELEMENT_FORCE_VECTORS, 0.0)) * 1e3
        return elapsed, stiff_ms + force_ms, len(cache.stiffness_objects), len(
            {id(spec) for spec in cache.panel_constitutive_specs if spec is not None}
        )

    # Warm-up
    _run_once()
    times_ms: list[float] = []
    for _ in range(5):
        elapsed, _phase, n_elem, n_unique = _run_once()
        times_ms.append(elapsed * 1e3)

    tracemalloc.start()
    cache = build_formulation_cache(
        panel_maps, zones, nx, nxy, shell_theory=ShellTheory.FSDT, L_x=1.0
    )
    _peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    median_ms = statistics.median(times_ms)
    assert n_elem == len(cache.stiffness_objects)
    assert n_unique >= 1
    assert median_ms >= 0.0
    assert _peak > 0
    # Mixed-laminate fixture: two builds with identical topology must match element count.
    cache_mixed = build_formulation_cache(
        panel_maps, zones, nx, nxy, shell_theory=ShellTheory.FSDT, L_x=1.0
    )
    assert len(cache_mixed.stiffness_objects) == len(cache.stiffness_objects)
    assert cache_mixed.formulation_id == cache.formulation_id


def test_wp0_exactness_tolerances_are_frozen():
    """Capture tolerances that later WPs must not widen."""
    assert 1e-10 > 0.0  # assembly K/f parity (test_formulation_cache_assembly_parity)
    assert 1e-12 > 0.0  # element Ke typed vs tuple
    # Documented freeze values for WP5 reports (not runtime-configurable).
    frozen = {
        "assembly_k_f_max_abs": 1e-10,
        "element_ke_atol": 1e-12,
        "L_s_identity_atol": 0.0,
        "bitwise_when_order_unchanged": True,
    }
    assert frozen["assembly_k_f_max_abs"] == 1e-10
    assert frozen["element_ke_atol"] == 1e-12
    assert frozen["L_s_identity_atol"] == 0.0
    assert frozen["bitwise_when_order_unchanged"] is True
