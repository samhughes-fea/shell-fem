"""MITC4 flat-shell element (Bathe & Dvorkin 1985).

Explicit public API (no star export): the element stiffness/recovery entrypoints
from :mod:`.element`, plus the formulation factory and native→canonical
projector. Import deep submodules only from within this worker.
"""

from .element import (
    Mitc4StiffnessContributions,
    assemble_mitc4_stiffness,
    clear_ke_template_cache,
    ke_template_cache_stats,
    mitc4_edge_resultants,
    mitc4_edge_shear_traction_integrated,
    mitc4_generalized_resultants_at,
    mitc4_geometric_stiffness,
    mitc4_resultants,
    mitc4_resultants_at,
    mitc4_shear_resultants,
    mitc4_stiffness,
    mitc4_stiffness_contributions,
)
from .factory import Mitc4ElementFactory
from .projector import Mitc4NativeToCanonicalProjector

__all__ = [
    "Mitc4ElementFactory",
    "Mitc4NativeToCanonicalProjector",
    "Mitc4StiffnessContributions",
    "assemble_mitc4_stiffness",
    "clear_ke_template_cache",
    "ke_template_cache_stats",
    "mitc4_edge_resultants",
    "mitc4_edge_shear_traction_integrated",
    "mitc4_generalized_resultants_at",
    "mitc4_geometric_stiffness",
    "mitc4_resultants",
    "mitc4_resultants_at",
    "mitc4_shear_resultants",
    "mitc4_stiffness",
    "mitc4_stiffness_contributions",
]
