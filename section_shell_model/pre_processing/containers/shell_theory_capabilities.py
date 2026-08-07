"""Shell theory enums, regime tables, and dispatch capability records (R6).

Split from ``containers.types`` so DTOs and theory-registration vocabulary
do not share one mega-module. ``containers.types`` re-exports these symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

class ProvenanceKind(str, Enum):
    """How a resultant value was obtained."""

    DERIVED = "derived"          # from section recovery (thin-wall mapping)
    PLACEHOLDER = "placeholder"  # not yet recovered; set to explicit default
    RESERVED = "reserved"        # field reserved for future FSDT / higher-order recovery
    MITC4 = "mitc4"              # recovered from MITC4 panel shell solve
    HSDT = "hsdt"                # recovered from native HSDT (8-DOF) panel shell solve
    CALCULIX_CCX = "calculix_ccx"  # recovered from external CalculiX (ccx) S4R solve


_TRUSTED_PROVENANCE_KINDS: frozenset[ProvenanceKind] = frozenset({
    ProvenanceKind.DERIVED,
    ProvenanceKind.MITC4,
    ProvenanceKind.HSDT,
    ProvenanceKind.CALCULIX_CCX,
})


def provenance_is_trusted(kind: ProvenanceKind) -> bool:
    """True when a resultant field was computed by a trusted shell/kernel path."""
    return kind in _TRUSTED_PROVENANCE_KINDS


def provenance_is_placeholder(kind: ProvenanceKind) -> bool:
    """True when a resultant is an explicit stub (must not govern failure indices)."""
    return kind is ProvenanceKind.PLACEHOLDER


class ShellAnalysisRegime(str, Enum):
    """Canonical shell analysis regime persisted on shell-owned station artifacts."""

    LINEAR = "linear"
    NONLINEAR = "nonlinear"


class ShellTheory(str, Enum):
    """Canonical shell theory selection persisted on shell-owned station artifacts."""

    FSDT = "fsdt"
    CALCULIX_CCX = "calculix_ccx"


SUPPORTED_SHELL_THEORY_BY_REGIME: dict[ShellAnalysisRegime, tuple[ShellTheory, ...]] = {
    # CALCULIX_CCX removed (Phase 6 CCX quarantine): it is an external,
    # opt-in validation backend (register_calculix_ccx_backend()), not a
    # production analysis theory a caller should be able to select through
    # the general regime/theory config validator. The enum value and its
    # SHELL_THEORY_CAPABILITIES row (dispatch-feature flags consulted
    # directly by the three dispatch seams) are unchanged — only this
    # regime-membership tuple, consulted by validate_shell_configuration(),
    # is tightened. No production ShellStationProvenance /
    # ShellStationBucklingBundle construction site passes
    # shell_theory=CALCULIX_CCX (verified at removal time).
    ShellAnalysisRegime.LINEAR: (ShellTheory.FSDT,),
    # NONLINEAR empty: geometric Newton is out of scope for this package
    # (not a scheduled milestone). Production uses the generalised linear
    # section BVP (contour-supported linear-static BVP + load channels).
    ShellAnalysisRegime.NONLINEAR: (),
}


EXPERIMENTAL_SHELL_THEORIES_BY_REGIME: dict[ShellAnalysisRegime, tuple[ShellTheory, ...]] = {
    ShellAnalysisRegime.LINEAR: (),
    ShellAnalysisRegime.NONLINEAR: (),
}


def supported_shell_theories(
    analysis_regime: ShellAnalysisRegime,
) -> tuple[ShellTheory, ...]:
    """Return supported shell theories for a given analysis regime."""

    return SUPPORTED_SHELL_THEORY_BY_REGIME.get(analysis_regime, ())


def experimental_shell_theories(
    analysis_regime: ShellAnalysisRegime,
) -> tuple[ShellTheory, ...]:
    """Return experimental shell theories tracked but not yet production-enabled."""

    return EXPERIMENTAL_SHELL_THEORIES_BY_REGIME.get(analysis_regime, ())


def validate_shell_configuration(
    analysis_regime: ShellAnalysisRegime,
    shell_theory: ShellTheory,
) -> None:
    """Raise ``ValueError`` when the shell regime/theory pair is unsupported."""

    supported = supported_shell_theories(analysis_regime)
    if shell_theory not in supported:
        supported_labels = ", ".join(theory.value for theory in supported) or "none"
        raise ValueError(
            "Unsupported shell configuration: "
            f"analysis_regime={analysis_regime.value!r}, shell_theory={shell_theory.value!r}. "
            f"Supported theories for regime {analysis_regime.value!r}: {supported_labels}."
        )


@dataclass(frozen=True)
class ShellTheoryCapability:
    """Single source of truth for per-theory dispatch traits.

    Consulted by every dispatch seam so that "which theories exist and what each
    supports" is defined in exactly one place:

    * the element-kernel resolver (:mod:`.element_library.kernel`, a sibling
      package under ``pre_processing``),
    * the global-solve backend registry (:mod:`..dispatch.shell_solve_registry`),
    * the linear-static section entry
      (``..simulation_runner.api.linear_static.run_section_linear_static_from_resultants``), and
    * the per-station dispatch
      (``..dispatch.station_theory_dispatch.dispatch_station_theory``).
    """

    theory: ShellTheory
    label: str
    provenance_kind: ProvenanceKind
    has_native_strip_kernel: bool
    supports_periodic: bool
    supports_station_dispatch: bool
    requires_external_work_dir: bool


SHELL_THEORY_CAPABILITIES: dict[ShellTheory, ShellTheoryCapability] = {
    ShellTheory.FSDT: ShellTheoryCapability(
        theory=ShellTheory.FSDT,
        label="native_mitc4_fsdt",
        provenance_kind=ProvenanceKind.MITC4,
        has_native_strip_kernel=True,
        supports_periodic=True,
        supports_station_dispatch=True,
        requires_external_work_dir=False,
    ),
    ShellTheory.CALCULIX_CCX: ShellTheoryCapability(
        theory=ShellTheory.CALCULIX_CCX,
        label="external_calculix_ccx",
        provenance_kind=ProvenanceKind.CALCULIX_CCX,
        has_native_strip_kernel=False,
        supports_periodic=True,
        supports_station_dispatch=True,
        requires_external_work_dir=True,
    ),
}


_CAPABILITY_FEATURE_ATTR: dict[str, str] = {
    "periodic": "supports_periodic",
    "station_dispatch": "supports_station_dispatch",
    "native_strip_kernel": "has_native_strip_kernel",
}


def resolve_shell_theory_capability(theory: ShellTheory) -> ShellTheoryCapability:
    """Return the dispatch capability record for ``theory`` (raises for unknown)."""
    try:
        return SHELL_THEORY_CAPABILITIES[theory]
    except KeyError:
        known = ", ".join(t.value for t in SHELL_THEORY_CAPABILITIES) or "none"
        raise ValueError(
            f"Unknown shell theory {theory!r}; registered theories: {known}."
        ) from None


def shell_theories_supporting(feature: str) -> tuple[ShellTheory, ...]:
    """Theories whose capability flag for ``feature`` is set (single dispatch seam)."""
    attr = _CAPABILITY_FEATURE_ATTR[feature]
    return tuple(t for t, c in SHELL_THEORY_CAPABILITIES.items() if getattr(c, attr))


def require_shell_theory_supports(
    theory: ShellTheory, *, feature: str
) -> ShellTheoryCapability:
    """Return the capability for ``theory`` or raise if it lacks ``feature``."""
    cap = resolve_shell_theory_capability(theory)
    if not getattr(cap, _CAPABILITY_FEATURE_ATTR[feature]):
        supported = ", ".join(t.value for t in shell_theories_supporting(feature)) or "none"
        raise ValueError(
            f"Shell theory {theory.value!r} does not support {feature!r}; "
            f"supported: {supported}."
        )
    return cap


