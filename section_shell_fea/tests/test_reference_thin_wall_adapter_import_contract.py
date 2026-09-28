"""WP6: FE reference_thin_wall adapter contracts and BSU parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_FE_ROOT = Path(__file__).resolve().parents[1]
_RTW = _FE_ROOT / "pre_processing" / "reference_thin_wall"


def _iter_prod_py(root: Path) -> list[Path]:
    skip = {"tests", "visualisation", "examples", "docs", "archive", "__pycache__"}
    out: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in skip for part in path.parts):
            continue
        out.append(path)
    return out


def test_reference_thin_wall_does_not_import_processing_or_solve() -> None:
    banned = (
        "processing.static",
        "processing.dynamic",
        "simulation_runner",
        "shell_solve_registry",
        "SectionThinWallModelRunner",
    )
    hits: list[str] = []
    for path in _iter_prod_py(_RTW):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                hits.append(f"{path.relative_to(_FE_ROOT).as_posix()}: {token}")
    assert not hits, "reference_thin_wall must stay pre-solve only:\n" + "\n".join(hits)


def test_loading_thin_wall_is_shim_only() -> None:
    shim = _FE_ROOT / "pre_processing" / "loading" / "thin_wall" / "__init__.py"
    text = shim.read_text(encoding="utf-8")
    assert "DeprecationWarning" in text
    assert "reference_thin_wall" in text


def test_omega_increment_parity_with_bsu_kernel() -> None:
    from blade_structure_utilities.geometry_utilities.section_thin_wall_model import (
        omega_increment as bsu_inc,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.reference_thin_wall.sectorial_warping import (
        omega_increment as fe_inc,
        omega_vertices_open_chain,
    )

    args = (0.0, 0.0, 1.0, 0.5, 0.1, -0.2)
    assert fe_inc(*args) == bsu_inc(*args)
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    omega = omega_vertices_open_chain(verts, 0.0, 0.0)
    expected = np.zeros(4)
    for i in range(3):
        expected[i + 1] = expected[i] + bsu_inc(
            verts[i, 0], verts[i, 1], verts[i + 1, 0], verts[i + 1, 1], 0.0, 0.0
        )
    np.testing.assert_allclose(omega, expected)


def test_exports_fe_adapter_products_only() -> None:
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing import (
        reference_thin_wall as rtw,
    )

    banned = {"build_wall_graph", "ThinWall", "WallGraph", "SectionThinWallModelRunner"}
    exported = set(rtw.__all__)
    assert not (exported & banned)
    assert "compute_section_vlasov" in exported
    assert "SectionVlasovResult" in exported
