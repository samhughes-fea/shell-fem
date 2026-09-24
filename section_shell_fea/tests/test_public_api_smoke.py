"""Public worker-API seam: package barrel and stdlib-safe package cwd."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import blade_structure_utilities.fem_utilities.section_shell_fea as pkg
from blade_structure_utilities.fem_utilities.section_shell_fea import api


_PKG = Path(pkg.__file__).resolve().parent


def test_api_barrel_reexports_worker_all():
    assert list(api.__all__) == list(pkg.__all__)
    assert api.SectionLinearStaticRunner is pkg.SectionLinearStaticRunner
    assert not (_PKG / "types").exists()


def test_package_cwd_does_not_shadow_stdlib_types():
    code = (
        "import types\n"
        "from types import SimpleNamespace\n"
        "SimpleNamespace(a=1)\n"
        "assert 'section_shell_fea' not in types.__file__.replace('\\\\', '/')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_PKG),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    assert result.returncode == 0, result.stderr
