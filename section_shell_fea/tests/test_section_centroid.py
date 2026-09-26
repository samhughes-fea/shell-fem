"""FEA-C-001: elastic centroid must not skip panels or return silent (0,0)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.validation.section_centroid import (
    compute_elastic_centroid_from_panels,
)


class _BadLam:
    pass


def test_elastic_centroid_raises_on_abd_failure():
    panel = SimpleNamespace(
        label="skin:a",
        lam=_BadLam(),
        nodes=np.array([[0.0, 0.0], [1.0, 0.0]]),
        s=np.array([0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="FEA-C-001"):
        compute_elastic_centroid_from_panels([panel])


def test_elastic_centroid_raises_on_short_geometry():
    class _OkLam:
        pass

    panel = SimpleNamespace(
        label="skin:a",
        lam=_OkLam(),
        nodes=np.array([[0.0, 0.0]]),
        s=np.array([0.0]),
    )

    def _fake_abd(_lam):
        abd = np.eye(6, dtype=float)
        abd[0, 0] = 1e8
        return abd, 0.01, 0.3, 70e9, 27e9

    import blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.constitutive.panel_abd as abd_mod

    prev = abd_mod.panel_abd_tuple_from_lam
    abd_mod.panel_abd_tuple_from_lam = _fake_abd  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="FEA-C-001"):
            compute_elastic_centroid_from_panels([panel])
    finally:
        abd_mod.panel_abd_tuple_from_lam = prev  # type: ignore[assignment]


def test_elastic_centroid_raises_on_empty_zones():
    with pytest.raises(ValueError, match="FEA-C-001"):
        compute_elastic_centroid_from_panels([])


def test_elastic_centroid_happy_path(monkeypatch):
    abd = np.zeros((6, 6), dtype=float)
    abd[0, 0] = 1e8

    def _fake_abd(_lam):
        return abd, 0.01, 0.3, 70e9, 27e9

    monkeypatch.setattr(
        "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.constitutive.panel_abd.panel_abd_tuple_from_lam",
        _fake_abd,
    )
    panel = SimpleNamespace(
        label="skin:a",
        lam=object(),
        nodes=np.array([[0.0, 0.0], [2.0, 0.0]]),
        s=np.array([0.0, 2.0]),
    )
    y_e, z_e = compute_elastic_centroid_from_panels([panel])
    assert abs(y_e - 1.0) < 1e-12
    assert abs(z_e - 0.0) < 1e-12
