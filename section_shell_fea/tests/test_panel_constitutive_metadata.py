"""Unit tests for _panel_constitutive_metadata's isotropic/orthotropic FI-kind
classification, including the is_isotropic_source marker path (the actual
production route, via materials_adapter's PanelLaminateAdapter)
alongside the legacy bare-IsotropicMaterial isinstance path (never constructed
in production, kept only for compatibility).
"""

from dataclasses import dataclass, field

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.formulation_factory import (
    _panel_constitutive_metadata,
)


@dataclass
class _FakePanelMap:
    s_nodes: np.ndarray = field(default_factory=lambda: np.array([0, 1]))


@dataclass
class _FakePanel:
    lam: object


@dataclass
class _IsotropicSourceLam:
    """Duck-types PanelLaminateAdapter's marker + yield strength."""
    is_isotropic_source: bool = True
    sigma_y_pa: float = 3.0e8
    sigma_uts_pa: float = 4.0e8
    sigma_ucs_pa: float = 3.5e8
    tau_us_pa: float = 2.0e8


@dataclass
class _OrthotropicLam:
    """A generic non-isotropic laminate stand-in (no is_isotropic_source)."""
    E: float = 6.0e10


def test_is_isotropic_source_marker_routes_to_isotropic_kind():
    panel_maps = [_FakePanelMap()]
    panels = [_FakePanel(lam=_IsotropicSourceLam(sigma_y_pa=3.0e8))]
    kinds, y, uts, ucs, tus = _panel_constitutive_metadata(panel_maps, zones)
    assert kinds == ["isotropic"]
    assert y[0] == pytest.approx(3.0e8)
    assert uts[0] == pytest.approx(4.0e8)
    assert ucs[0] == pytest.approx(3.5e8)
    assert tus[0] == pytest.approx(2.0e8)


def test_orthotropic_laminate_routes_to_orthotropic_kind():
    panel_maps = [_FakePanelMap()]
    panels = [_FakePanel(lam=_OrthotropicLam())]
    kinds, y, uts, ucs, tus = _panel_constitutive_metadata(panel_maps, zones)
    assert kinds == ["orthotropic"]
    assert np.isnan(y[0])
    assert np.isnan(uts[0])


def test_empty_panel_defaults_to_orthotropic():
    panel_maps = [_FakePanelMap(s_nodes=np.array([]))]
    panels = [_FakePanel(lam=_IsotropicSourceLam())]
    kinds, y, uts, ucs, tus = _panel_constitutive_metadata(panel_maps, zones)
    assert kinds == ["orthotropic"]
    assert np.isnan(y[0])


def test_mixed_panels_classified_independently():
    panel_maps = [_FakePanelMap(), _FakePanelMap()]
    panels = [
        _FakePanel(lam=_IsotropicSourceLam(sigma_y_pa=6.5e8)),
        _FakePanel(lam=_OrthotropicLam()),
    ]
    kinds, y, uts, ucs, tus = _panel_constitutive_metadata(panel_maps, zones)
    assert kinds == ["isotropic", "orthotropic"]
    assert y[0] == pytest.approx(6.5e8)
    assert np.isnan(y[1])


def test_yield_strength_preferred_over_material_sigma_allow_bridge():
    @dataclass
    class _Both:
        is_isotropic_source: bool = True
        yield_strength: float = 3.0e8
        sigma_allow: float = 1.0e8

    kinds, y, *_ = _panel_constitutive_metadata([_FakePanelMap()], [_FakePanel(lam=_Both())])
    assert kinds == ["isotropic"]
    assert y[0] == pytest.approx(3.0e8)


def test_material_sigma_allow_bridge_when_yield_missing():
    @dataclass
    class _LegacyMat:
        is_isotropic_source: bool = True
        sigma_allow: float = 2.0e8

    kinds, y, *_ = _panel_constitutive_metadata([_FakePanelMap()], [_FakePanel(lam=_LegacyMat())])
    assert kinds == ["isotropic"]
    assert y[0] == pytest.approx(2.0e8)
