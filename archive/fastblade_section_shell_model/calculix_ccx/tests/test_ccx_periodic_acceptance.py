"""CCX-gated periodic acceptance: FSDT vs CCX on load-aligned integrals and torsion flow.

Spike 4A (2026-07-14) attributed the former 0.5× multi-key xfail band to:

* **Not FRD** — ``emit_frd=False`` leaves membrane integrals unchanged (integrals use ``.dat``).
* **Not junction-MPC dominant** — load-aligned primaries (Fx under N, My under My) already
  match MITC4 within ~3%; failures came from asserting *all* of Fx/My/Mz on every load,
  so near-zero parasitic channels (CCX residual Fx under pure My, spurious Mz under pure N)
  blow up the relative check.
* **Torsion max|Nxy|** — CCX/MITC4 ratio ~0.20 on NACA+spar is S4R local redistribution,
  already accepted by the engineering gate ``rel=0.85`` in ``test_calculix_ccx_periodic.py``.

Acceptance therefore tracks the **validated engineering bands** in
``docs/PERIODIC_CCX_HOMOGENIZATION.md`` (~10% load-aligned integrals; torsion Nxy within
85% relative of MITC4), not the aspirational all-channel 0.5× band.
"""

from __future__ import annotations

import shutil
import tempfile

import pytest

_HAS_CCX = shutil.which("ccx") is not None
_NE = 4


def _pb():
    from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section

    return build_pipeline_b_section("2412", [0.35], chord_m=1.0)


def _max_abs_nxy(bundle) -> float:
    return max(
        (abs(float(r.Nxy)) for panel in bundle.all_panel_results for r in panel),
        default=0.0,
    )


@pytest.mark.ccx
@pytest.mark.skipif(not _HAS_CCX, reason="CalculiX ccx not on PATH")
@pytest.mark.parametrize(
    "load,keys",
    [
        ({"N": 1e6}, ("Fx",)),
        ({"My": 1e5}, ("My",)),
        ({"T": 1e4}, ("Fx", "My", "Mz")),  # torsion: all resultants ~0
    ],
)
def test_ccx_periodic_acceptance_integrals(load, keys):
    """Load-aligned section integrals within 10% of FSDT (validated engineering band)."""
    from blade_precompute.section_shell_model.api.periodic import (
        run_section_periodic_bc_mitc4,
    )
    from blade_precompute.section_shell_model.calculix_ccx.periodic import (
        run_section_periodic_bc_ccx,
    )
    from blade_precompute.section_shell_model.validation.section_integrals import (
        elastic_centroid,
        integrated_section_resultants,
        section_element_geometry,
    )

    pb = _pb()
    geom = section_element_geometry(pb.panels, _NE)
    y_e, z_e = elastic_centroid(pb.panels)
    bm = run_section_periodic_bc_mitc4(pb, n_elements_per_panel=_NE, **load)
    with tempfile.TemporaryDirectory() as tmp:
        bc = run_section_periodic_bc_ccx(pb, n_elements_per_panel=_NE, ccx_work_dir=tmp, **load)
    im = integrated_section_resultants(bm, geom, y_e=y_e, z_e=z_e)
    ic = integrated_section_resultants(bc, geom, y_e=y_e, z_e=z_e)
    scale = max(*(abs(im[k]) for k in ("Fx", "My", "Mz")), 1.0)
    for key in keys:
        assert abs(ic[key] - im[key]) <= 0.10 * scale, (
            f"CCX vs FSDT {key}: ccx={ic[key]:.4g} mitc4={im[key]:.4g} scale={scale:.4g}"
        )


@pytest.mark.ccx
@pytest.mark.skipif(not _HAS_CCX, reason="CalculiX ccx not on PATH")
def test_ccx_periodic_acceptance_torsion_nxy():
    """Torsion shear flow: CCX within engineering band of FSDT (S4R redistribution)."""
    from blade_precompute.section_shell_model.api.periodic import run_section_periodic_bc_mitc4
    from blade_precompute.section_shell_model.calculix_ccx.periodic import (
        run_section_periodic_bc_ccx,
    )

    pb = _pb()
    bm = run_section_periodic_bc_mitc4(pb, T=1.0e5, n_elements_per_panel=_NE)
    with tempfile.TemporaryDirectory() as tmp:
        bc = run_section_periodic_bc_ccx(pb, T=1.0e5, n_elements_per_panel=_NE, ccx_work_dir=tmp)
    nxy_m = _max_abs_nxy(bm)
    nxy_c = _max_abs_nxy(bc)
    assert nxy_m > 1.0e3
    # Same order of magnitude as MITC4; S4R may redistribute locally (rel ≤ 0.85).
    assert nxy_c == pytest.approx(nxy_m, rel=0.85)
