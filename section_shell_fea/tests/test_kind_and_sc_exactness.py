"""FEA-SC-001 / FEA-L-005 / FEA-CCX-003 exactness unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.shear_centre_loading_axis import (
    shear_centre_about_yz_from_panel_topology,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.equilibrium_checks import (
    _is_web_label,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.cap_tie.geometry import (
    kind_of,
)


def test_kind_of_prefix_not_substring():
    assert kind_of("web:spar_1") == "web"
    assert kind_of("skin:webbed_bay") == "skin"
    assert not _is_web_label("skin:webbed_bay")
    assert _is_web_label("web:spar_1")


def test_shear_centre_refuses_silent_thickness_default():
    mid = SimpleNamespace(
        kind="skin",
        label="skin:a",
        thickness_m=0.0,
        points=np.column_stack(
            [np.linspace(0.0, 1.0, 5), np.zeros(5)]
        ),
        midline_twisted_pitch_axis_blade_frame=None,
    )
    topo = SimpleNamespace(shell_inputs=SimpleNamespace(midlines=(mid,)))
    with pytest.raises(ValueError, match="FEA-SC-001"):
        shear_centre_about_yz_from_panel_topology(topo)


def test_validation_interlaminar_refuses_silent_thickness():
    from blade_structure_utilities.fem_utilities.section_shell_fea.validation.section_integrals import (
        multistation_equilibrium_interlaminar,
    )

    pb = SimpleNamespace(
        zones=[SimpleNamespace(lam=SimpleNamespace(t=0.0))],
        shell_inputs=SimpleNamespace(midlines=()),
    )
    with pytest.raises(ValueError, match="FEA-V-001"):
        multistation_equilibrium_interlaminar(
            pb, solver=lambda *a, **k: None, My0=0.0, Vz=1.0, dz=0.1
        )
