"""Unit tests for _panel_constitutive_metadata's isotropic/orthotropic FI-kind
classification, including the is_isotropic_source marker path (the actual
production route, via materials_adapter's PanelLaminateAdapter)
alongside the legacy bare-IsotropicMaterial isinstance path (never constructed
in production, kept only for compatibility).
"""

from dataclasses import dataclass, field

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.element_library.formulation_factory import (
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
    """Duck-types PanelLaminateAdapter's marker + sigma allowable."""
    is_isotropic_source: bool = True
    sigma_allow_pa: float = 3.0e8


@dataclass
class _OrthotropicLam:
    """A generic non-isotropic laminate stand-in (no is_isotropic_source)."""
    E: float = 6.0e10


def test_is_isotropic_source_marker_routes_to_isotropic_kind():
    panel_maps = [_FakePanelMap()]
    panels = [_FakePanel(lam=_IsotropicSourceLam(sigma_allow_pa=3.0e8))]
    kinds, allows = _panel_constitutive_metadata(panel_maps, panels)
    assert kinds == ["isotropic"]
    assert allows[0] == pytest.approx(3.0e8)


def test_orthotropic_laminate_routes_to_orthotropic_kind():
    panel_maps = [_FakePanelMap()]
    panels = [_FakePanel(lam=_OrthotropicLam())]
    kinds, allows = _panel_constitutive_metadata(panel_maps, panels)
    assert kinds == ["orthotropic"]
    assert np.isnan(allows[0])


def test_empty_panel_defaults_to_orthotropic():
    panel_maps = [_FakePanelMap(s_nodes=np.array([]))]
    panels = [_FakePanel(lam=_IsotropicSourceLam())]
    kinds, allows = _panel_constitutive_metadata(panel_maps, panels)
    assert kinds == ["orthotropic"]
    assert np.isnan(allows[0])


def test_mixed_panels_classified_independently():
    panel_maps = [_FakePanelMap(), _FakePanelMap()]
    panels = [
        _FakePanel(lam=_IsotropicSourceLam(sigma_allow_pa=6.5e8)),
        _FakePanel(lam=_OrthotropicLam()),
    ]
    kinds, allows = _panel_constitutive_metadata(panel_maps, panels)
    assert kinds == ["isotropic", "orthotropic"]
    assert allows[0] == pytest.approx(6.5e8)
    assert np.isnan(allows[1])
