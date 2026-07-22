"""The policy interface: the benchmark's adoption surface.

A policy implements `evict`; `on_access`, `on_hint`, and `place` have no-op
defaults, so the minimal policy is a few lines (docs/policy-interface.md,
"small enough to implement in an afternoon"). The harness binds a read-only
cache view once via `bind`, so `evict` keeps the documented
`evict(needed, now)` signature while still seeing the resident set.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BlockMeta:
    """What a policy sees about a resident block. Mirrors miniserve's BlockMeta
    seam; kind and recompute_cost come from the trace and cost model."""

    block_id: int
    kind: str  # system_prompt | history | tool_output | reasoning
    size_tokens: int
    recompute_cost: float  # cost to reconstruct if evicted then re-accessed
    last_access_ms: int
    access_count: int = 0
    tier: str = "gpu"


class CacheView:
    """Read-only view the harness hands the policy: the EVICTABLE resident
    blocks. Blocks the current request needs (its working set) are protected
    and excluded, because a request's attention runs over its whole prefix at
    once, so evicting a block it still needs is never valid. A policy therefore
    physically cannot choose a protected block."""

    def __init__(self, resident: dict[int, BlockMeta]):
        self._resident = resident
        self._protected: frozenset[int] = frozenset()

    def _set_protected(self, protected: frozenset[int]) -> None:
        """Harness-only: called once per request before eviction."""
        self._protected = protected

    def resident(self) -> dict[int, BlockMeta]:
        if not self._protected:
            return dict(self._resident)
        return {b: m for b, m in self._resident.items() if b not in self._protected}

    def resident_tokens(self) -> int:
        return sum(
            m.size_tokens
            for b, m in self._resident.items()
            if b not in self._protected
        )


class Policy(ABC):
    """Implement evict(); override the others only if your policy uses them."""

    def bind(self, cache: CacheView) -> None:
        """Called once by the harness before replay. Stores the cache view."""
        self.cache = cache

    @abstractmethod
    def evict(self, needed_tokens: int, now_ms: int) -> list[int]:
        """Return block_ids to evict so at least needed_tokens free up. Choose
        from self.cache.resident()."""

    def on_access(self, block_id: int, meta: BlockMeta, now_ms: int) -> None:  # noqa: B027
        """A resident block was accessed (a hit). Update recency structures.
        Optional override; no-op by default."""

    def on_hint(self, event: dict, now_ms: int) -> None:  # noqa: B027
        """A lifecycle hint (span_close, compaction, subagent_terminate).
        Optional override; ignored by inference-only policies."""

    def place(self, meta: BlockMeta) -> str:
        """Tier for a newly admitted block. Default keeps everything on gpu;
        offloading policies return a cheaper tier."""
        return "gpu"
