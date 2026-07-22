"""Package-local default wall laminates for topology when callers omit overrides.

Mirrors the historical SKIN/WEB/CAP isotropic defaults (E/nu/t with a
[0/90/90/0] ply pattern) so geometry bootstrap does not need a vendor
``sys.path`` bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

from blade_precompute.materials.laminate import ClptPlyRow, clpt_ply_row

_DEFAULT_NU = 0.35
_DEFAULT_N_PLIES = 4
_ANGLE_PATTERN_DEG = (0.0, 90.0, 90.0, 0.0)


@dataclass(frozen=True)
class WallLaminate:
    """Thin-wall isotropic laminate with ``build_plies()`` for ABD recovery."""

    E: float
    t: float
    name: str = ""
    nu: float = _DEFAULT_NU
    n_plies: int = _DEFAULT_N_PLIES

    def build_plies(self) -> list[ClptPlyRow]:
        """Representative [0/90/90/0] isotropic stack (matches vendor defaults)."""
        n = max(int(self.n_plies), 1)
        t_ply = float(self.t) / float(n)
        g12 = float(self.E) / (2.0 * (1.0 + float(self.nu)))
        rows: list[ClptPlyRow] = []
        for i in range(n):
            theta = _ANGLE_PATTERN_DEG[i % len(_ANGLE_PATTERN_DEG)]
            rows.append(
                clpt_ply_row(
                    E1=float(self.E),
                    E2=float(self.E),
                    G12=g12,
                    nu12=float(self.nu),
                    theta_deg=theta,
                    t=t_ply,
                )
            )
        return rows


def default_skin_web_cap_laminates() -> tuple[WallLaminate, WallLaminate, WallLaminate]:
    """Return (skin, web, cap) defaults matching historical demo laminate values."""
    skin = WallLaminate(E=45e9, t=0.001, name="Skin CFRP", n_plies=4)
    web = WallLaminate(E=43e9, t=0.001, name="Shear web +-45 CFRP", n_plies=4)
    cap = WallLaminate(E=120e9, t=0.020, name="Spar cap UD CFRP", n_plies=4)
    return skin, web, cap


__all__ = ["WallLaminate", "default_skin_web_cap_laminates"]
