"""kv-policy-core: the neutral, engine-agnostic policy interface.

This package is the standard the project proposes: it depends on nothing (not the
benchmark, not any engine), so miniserve, agentic-kv-bench, and future adapters
(vLLM/LMCache) can all depend on IT without coupling to each other. That is what
lets one policy implementation run in the simulator, the instrument, and a real
engine unchanged (offline/online parity).

Contents: BlockMeta (what a policy sees), CacheView (the read-only evictable set),
Policy (implement evict; the rest default), Decision (the general action type; L2
uses Evict only), and Port (the seam an engine implements so a policy can drive it).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class BlockMeta:
    """What a policy sees about a resident block. Mirrors miniserve's BlockMeta
    seam; kind and recompute_cost come from the trace and cost model."""

    block_id: object
    kind: str  # system_prompt | history | tool_output | reasoning
    size_tokens: int
    recompute_cost: float  # cost to reconstruct if evicted then re-accessed
    last_access_ms: int
    access_count: int = 0
    tier: str = "gpu"


class CacheView:
    """Read-only view the port hands the policy: the EVICTABLE resident blocks.
    Blocks the current request needs (its working set) are protected and excluded,
    because a request's attention runs over its whole prefix at once, so evicting a
    block it still needs is never valid. A policy therefore physically cannot choose
    a protected block."""

    def __init__(self, resident: dict):
        self._resident = resident
        self._protected: frozenset = frozenset()

    def _set_protected(self, protected: frozenset) -> None:
        """Port-only: called once per request before eviction."""
        self._protected = protected

    def resident(self) -> dict:
        if not self._protected:
            return dict(self._resident)
        return {b: m for b, m in self._resident.items() if b not in self._protected}

    def resident_tokens(self) -> int:
        return sum(
            m.size_tokens for b, m in self._resident.items() if b not in self._protected
        )


@dataclass(frozen=True)
class Evict:
    """Drop this block. The only Decision L2 uses."""

    block_id: object
    reason: str = ""


# Reserved for L4, so the seam does not break when offload/refuse arrive:
#   Move(block_id, to_tier), RecomputeLater(block_id), Refuse(request_id), Keep(block_id)
# L2's action space is {Evict}; the type is already the general one.
Decision = Evict


class Policy(ABC):
    """Implement evict(); override the others only if your policy uses them.

    decide() is the general interface (returns Decisions); evict() is the L2
    restriction, and the default decide() wraps it, so trivial policies stay
    trivial while the type is ready for L4."""

    def bind(self, cache: CacheView) -> None:
        """Called once before replay/serving. Stores the cache view."""
        self.cache = cache

    @abstractmethod
    def evict(self, needed_tokens: int, now_ms: int) -> list:
        """Return block_ids to evict so at least needed_tokens free up. Choose
        from self.cache.resident()."""

    def decide(self, needed_tokens: int, now_ms: int) -> list:
        """General action interface; defaults to wrapping evict()."""
        return [Evict(b) for b in self.evict(needed_tokens, now_ms)]

    def on_access(self, block_id, meta: BlockMeta, now_ms: int) -> None:  # noqa: B027
        """A resident block was accessed (a hit). Update recency structures.
        Optional override; no-op by default."""

    def on_hint(self, event: dict, now_ms: int) -> None:  # noqa: B027
        """A lifecycle hint (retire, compaction, session_close, ...).
        Optional override; ignored by inference-only policies."""

    def place(self, meta: BlockMeta) -> str:
        """Tier for a newly admitted block. Default keeps everything on gpu;
        offloading policies return a cheaper tier."""
        return "gpu"

    def maintain(self, now_ms: int) -> list:
        """Proactive-eviction opportunity, called once per request before
        admission. Return block_ids to evict now; default none."""
        return []


@runtime_checkable
class Port(Protocol):
    """The engine-agnostic seam a policy is driven against. `SimPort` (this repo)
    and, from step 2, `MiniservePort` (the real block pool) implement it. The policy
    object is identical across both — that is the offline/online-parity mechanism.

    `mode` is 'enforce' (decisions enacted) or 'advisory' (decisions logged/scored,
    the engine's own path runs) — shadow-before-enforce."""

    mode: str

    def bind_policy(self, policy: Policy) -> None: ...
    def set_protected(self, working: frozenset) -> None: ...
    def resident(self) -> dict: ...
    def get(self, block_id): ...
    def admit(self, meta: BlockMeta) -> None: ...
    def free_to_fit(self, needed_tokens: int, now_ms: int, protected: frozenset) -> None: ...
    def now_ms(self) -> int: ...
