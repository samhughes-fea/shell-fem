"""Generic per-build stiffness-template cache (formulation-agnostic).

One :class:`ShellFormulationTemplateCache` is created per
``build_formulation_cache`` call and handed to a formulation factory that opts in
via a ``template_cache=`` keyword. The cache stores whatever opaque built product
(dense ``Ke``, operator handles, …) a formulation chooses to reuse, keyed by an
opaque hashable identity the formulation derives (see
``registry.TemplateKeyStrategy``).

Exactness contract (WP0): keys are exact — never rounded — so production reuse is
bitwise-identical. An incomplete identity (``key is None``) is *refused*: it is a
miss and is never stored, so uncertain identities recompute rather than alias.

No MITC4 (or any formulation) types appear here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable


@dataclass(frozen=True, slots=True)
class TemplateCacheStats:
    """Immutable snapshot of one build's template-cache activity."""

    hits: int
    misses: int
    refusals: int
    unique: int


class ShellFormulationTemplateCache:
    """Per-build get/put store with hit/miss/refusal/unique accounting."""

    __slots__ = ("_store", "_hits", "_misses", "_refusals")

    def __init__(self) -> None:
        self._store: dict[Hashable, Any] = {}
        self._hits = 0
        self._misses = 0
        self._refusals = 0

    def get(self, key: Hashable | None) -> Any | None:
        """Return the cached value for ``key`` or ``None`` (miss/refusal).

        A ``None`` key is an incomplete identity: refused (never a hit).
        """
        if key is None:
            self._refusals += 1
            return None
        if key in self._store:
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def put(self, key: Hashable | None, value: Any) -> None:
        """Store ``value`` under ``key``. Incomplete identities are not stored."""
        if key is None:
            return
        self._store[key] = value

    def get_or_build(
        self, key: Hashable | None, build: Callable[[], Any]
    ) -> Any:
        """Return cached value for ``key`` or build, store (if keyed), and return.

        Refused (``None``) keys always rebuild and are never stored.
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        value = build()
        self.put(key, value)
        return value

    def stats(self) -> TemplateCacheStats:
        return TemplateCacheStats(
            hits=self._hits,
            misses=self._misses,
            refusals=self._refusals,
            unique=len(self._store),
        )

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0
        self._refusals = 0


__all__ = [
    "ShellFormulationTemplateCache",
    "TemplateCacheStats",
]
