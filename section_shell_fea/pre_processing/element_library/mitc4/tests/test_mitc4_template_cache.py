"""WP3 — MITC4 formulation-owned template caching (exact keys, within-build reuse)."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.mitc4.element import (
    clear_ke_template_cache,
    ke_template_cache_stats,
    mitc4_stiffness,
    mitc4_stiffness_contributions,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.mitc4.mitc4_quad4_element import (
    Mitc4ElementGeometry,
    Mitc4ElementLoads,
    Mitc4ElementMaterial,
    Mitc4Quad4Element,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.template_cache import (
    ShellFormulationTemplateCache,
)


def _abd() -> np.ndarray:
    a = np.array([[120.0, 20.0, 5.0], [20.0, 90.0, 3.0], [5.0, 3.0, 40.0]])
    d = np.array([[12.0, 2.0, 0.5], [2.0, 9.0, 0.3], [0.5, 0.3, 4.0]])
    z = np.zeros((3, 3))
    return np.block([[a, z], [z, d]])


def _element(*, L_s: float, template_cache=None, abd=None) -> Mitc4Quad4Element:
    return Mitc4Quad4Element(
        element_id=0,
        panel_index=0,
        local_element_index=0,
        geometry=Mitc4ElementGeometry(
            L_s=L_s, L_s_donnell=L_s, L_x=1.0, local_nodes=(0, 1, 2, 3)
        ),
        material=Mitc4ElementMaterial(
            ABD=_abd() if abd is None else abd, thickness=2.0e-3, G_eff=5.0e9
        ),
        loads=Mitc4ElementLoads(nx_avg=0.0, nxy_avg=0.0, kappa_avg=None),
        template_cache=template_cache,
    )


def test_per_build_cache_reuses_ke_for_identical_elements_bitwise():
    cache = ShellFormulationTemplateCache()
    a = _element(L_s=0.1, template_cache=cache).element_stiffness_matrix()
    b = _element(L_s=0.1, template_cache=cache).element_stiffness_matrix()
    # Reuse: identical key returns the very same array object (operators reused too).
    assert a.K_e is b.K_e
    # Bitwise-identical to a fresh no-cache assembly.
    fresh = _element(L_s=0.1, template_cache=None).element_stiffness_matrix()
    assert np.array_equal(a.K_e, fresh.K_e)
    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.unique == 1


def test_per_build_cache_distinguishes_geometry():
    cache = ShellFormulationTemplateCache()
    a = _element(L_s=0.1, template_cache=cache).element_stiffness_matrix()
    b = _element(L_s=0.2, template_cache=cache).element_stiffness_matrix()
    assert a.K_e is not b.K_e
    assert cache.stats().unique == 2


def test_per_build_cache_exact_no_rounding_collision():
    """A one-ULP length difference is a distinct template (no round() collision)."""
    cache = ShellFormulationTemplateCache()
    L0 = 0.1
    L1 = float(np.nextafter(L0, L0 + 1.0))
    assert L0 != L1
    _element(L_s=L0, template_cache=cache).element_stiffness_matrix()
    _element(L_s=L1, template_cache=cache).element_stiffness_matrix()
    assert cache.stats().unique == 2
    assert cache.stats().hits == 0


def test_module_compat_cache_exact_keys_stats_and_clear():
    clear_ke_template_cache()
    stats0 = ke_template_cache_stats()
    assert stats0 == {"hits": 0, "misses": 0, "unique": 0}

    abd = _abd()
    k1 = mitc4_stiffness(0.1, 1.0, abd, 2.0e-3, G_eff=5.0e9)
    k2 = mitc4_stiffness(0.1, 1.0, abd, 2.0e-3, G_eff=5.0e9)
    assert np.array_equal(k1, k2)
    stats = ke_template_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["unique"] == 1

    # One-ULP difference must miss (exact float bits, no rounding).
    mitc4_stiffness(float(np.nextafter(0.1, 1.0)), 1.0, abd, 2.0e-3, G_eff=5.0e9)
    assert ke_template_cache_stats()["misses"] == 2
    assert ke_template_cache_stats()["unique"] == 2

    # Compat cache result equals the direct contribution assembly (bitwise).
    ref = mitc4_stiffness_contributions(0.1, 1.0, abd, 2.0e-3, G_eff=5.0e9).K_e
    assert np.array_equal(k1, ref)
    clear_ke_template_cache()
    assert ke_template_cache_stats() == {"hits": 0, "misses": 0, "unique": 0}
