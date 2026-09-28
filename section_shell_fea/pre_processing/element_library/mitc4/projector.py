"""Native MITC4 products → canonical dense result layouts."""

from __future__ import annotations

from ..contracts import CANONICAL_RESULTANT_NAMES, CANONICAL_STRESS_NAMES, mitc4_fsdt_descriptor


class Mitc4NativeToCanonicalProjector:
    """MITC4-FSDT projector. Frame rotations are delegated to coordinate_frames."""

    formulation_id = "MITC4-FSDT"
    canonical_layout_version = "v1"

    def __init__(self) -> None:
        self.descriptor = mitc4_fsdt_descriptor()

    def project_resultants(
        self, native: dict[str, float]
    ) -> tuple[tuple[float, ...], tuple[str, ...]]:
        values = []
        caps = []
        declared = self.descriptor.generalized.capabilities
        for name in CANONICAL_RESULTANT_NAMES:
            values.append(float(native.get(name, 0.0)))
            caps.append(str(declared[name]))
        return tuple(values), tuple(caps)

    def project_material_stress(
        self,
        sigma11: float,
        sigma22: float,
        tau12: float,
        *,
        sigma33: float = 0.0,
        tau23: float = 0.0,
        tau13: float = 0.0,
    ) -> tuple[tuple[float, ...], tuple[str, ...]]:
        raw = {
            "sigma11": float(sigma11),
            "sigma22": float(sigma22),
            "sigma33": 0.0 if sigma33 == 0.0 else 0.0,
            "tau23": float(tau23),
            "tau13": float(tau13),
            "tau12": float(tau12),
        }
        # MITC4 never models sigma33; force the weak-form zero regardless of caller.
        raw["sigma33"] = 0.0
        caps = tuple(
            str(self.descriptor.stress_capabilities[name])
            for name in CANONICAL_STRESS_NAMES
        )
        values = tuple(raw[name] for name in CANONICAL_STRESS_NAMES)
        return values, caps


__all__ = ["Mitc4NativeToCanonicalProjector"]
