"""FEA-JN-001 / FEA-T-001 / FEA-BK-001 exactness unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.formulation_factory import (
    _isotropic_to_laminate_definition,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.connected_panel_topology import (
    _validate_producer_clusters,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_buckling_runner import (
    panel_inputs_from_nested_resultants,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.recovery_types import (
    ShellPanelResultants,
)


def test_validate_producer_clusters_covers_endpoints():
    endpoints = [
        (0, "start", np.array([0.0, 0.0])),
        (0, "end", np.array([1.0, 0.0])),
        (1, "start", np.array([1.0, 0.0])),
        (1, "end", np.array([1.0, 1.0])),
    ]
    clusters = [
        [(0, "start", np.array([0.0, 0.0]))],
        [(0, "end", np.array([1.0, 0.0])), (1, "start", np.array([1.0, 0.0]))],
        [(1, "end", np.array([1.0, 1.0]))],
    ]
    _validate_producer_clusters(clusters, endpoints)


def test_validate_producer_clusters_refuses_missing():
    endpoints = [
        (0, "start", np.array([0.0, 0.0])),
        (0, "end", np.array([1.0, 0.0])),
    ]
    clusters = [[(0, "start", np.array([0.0, 0.0]))]]
    with pytest.raises(ValueError, match="FEA-JN-001"):
        _validate_producer_clusters(clusters, endpoints)


def test_isotropic_refuses_silent_thickness():
    mat = SimpleNamespace(E=70e9, nu=0.3, rho=1800.0, name="iso")
    zone = SimpleNamespace(t=0.0, lam=SimpleNamespace(t=0.0), label="skin:a")
    with pytest.raises(ValueError, match="FEA-T-001"):
        _isotropic_to_laminate_definition(mat, zone)


def test_buckling_inputs_refuse_silent_geometry():
    row = ShellPanelResultants(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, thickness_m=0.0)
    with pytest.raises(ValueError, match="FEA-BK-001"):
        panel_inputs_from_nested_resultants([[row]], panel_widths_m=[0.5])
    row2 = ShellPanelResultants(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, thickness_m=0.005)
    with pytest.raises(ValueError, match="FEA-BK-001"):
        panel_inputs_from_nested_resultants([[row2]])
