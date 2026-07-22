"""Coverage for the CalculiX ccx strip-export + orientation helpers.

Preserves the strip-deck / ``quad_coords_from_strip`` coverage that used to live in the
deleted native-S4R kernel tests; the native pure-Python S4R element has been removed in favour
of the external ``ccx`` binary backend under ``element_library/calculix_ccx``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from blade_precompute.section_shell_model.calculix_ccx.orientation import (
    quad_coords_from_strip,
)
from blade_precompute.section_shell_model.calculix_ccx.strip_export import (
    export_rectangular_strip_inp,
    strip_coords_for_panel,
)


def test_quad_coords_from_strip_layout():
    s_nodes, elements, yz_nodes = strip_coords_for_panel(2.0, 4)
    coords = quad_coords_from_strip(yz_nodes, elements[0], l_x=1.0, n_contour=len(s_nodes))
    assert coords.shape == (4, 3)
    # Bottom layer sits at x=0, top layer at x=l_x.
    assert np.isclose(coords[0, 0], 0.0)
    assert np.isclose(coords[3, 0], 1.0)


def test_export_rectangular_strip_inp_writes_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = export_rectangular_strip_inp(Path(tmp) / "strip.inp")
        text = path.read_text(encoding="utf-8")
        assert "*ELEMENT, TYPE=S4R" in text
        assert "*SHELL SECTION" in text


@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX ccx not on PATH")
def test_ccx_oracle_strip_runs():
    with tempfile.TemporaryDirectory() as tmp:
        inp = export_rectangular_strip_inp(Path(tmp) / "strip.inp", nx_traction=500.0)
        subprocess.run(["ccx", str(inp.stem)], cwd=tmp, check=True, capture_output=True)
        frd = Path(tmp) / f"{inp.stem}.frd"
        assert frd.exists() or (Path(tmp) / f"{inp.stem}.dat").exists()
