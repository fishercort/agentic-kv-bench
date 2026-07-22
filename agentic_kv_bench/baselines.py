"""Bundled baseline policies, reimplemented from the literature per
docs/policy-interface.md. Fidelity matters (the adoption pitch to the papers'
authors is a faithful baseline), so each policy cites its source and rule, and
notes any simplification forced by the simulator's scope.

LRU is the floor. IdleTTL is a naive strawman (NOT Continuum). The ladder adds
GDSF, WA-LRU (SAGA), retired-cache lifecycle, and a faithful Continuum/CacheTTL
(gap-aware protection, hint-consuming) once the hint interface is wired.
"""

from agentic_kv_bench.policy import Policy


def _lru_victims(resident, needed_tokens: int) -> list:
    by_recency = sorted(resident.values(), key=lambda m: m.last_access_ms)
    victims, freed = [], 0
    for m in by_recency:
        victims.append(m.block_id)
        freed += m.size_tokens
        if freed >= needed_tokens:
            break
    return victims


class LRU(Policy):
    """Evict least-recently-accessed blocks. The floor: no cost awareness, no
    lifecycle awareness. Reads recency straight off the resident view, so it
    holds zero shadow state."""

    def evict(self, needed_tokens: int, now_ms: int) -> list:
        return _lru_victims(self.cache.resident(), needed_tokens)


class IdleTTL(Policy):
    """NAIVE idle-time expiry: a block idle longer than a fixed TTL is
    proactively evicted; under forced pressure, LRU.

    This is NOT Continuum / CacheTTL, and must never be labeled as such.
    Continuum's mechanism is gap-aware PROTECTION: it keeps KV alive through
    predicted tool-call gaps, assigning TTLs from workflow knowledge precisely
    so the cache survives the idle windows a naive evictor would harvest. In
    other words, naive idle-time eviction is Continuum's motivating
    counterexample, not Continuum. A faithful Continuum is a hint-consuming
    policy (it needs a predicted-next-turn signal) and lands as its own rung
    once the hint interface is wired; see docs/policy-interface.md.

    IdleTTL is included as a legitimate strawman: the mechanism lower bound
    that quantifies how badly a pure idle-time signal misleads on this
    workload. It confirms Continuum's premise; it does not test Continuum's
    contribution."""

    def __init__(self, ttl_ms: float = 60_000.0):
        self.ttl_ms = ttl_ms

    def maintain(self, now_ms: int) -> list:
        return [
            bid
            for bid, m in self.cache.resident().items()
            if now_ms - m.last_access_ms > self.ttl_ms
        ]

    def evict(self, needed_tokens: int, now_ms: int) -> list:
        return _lru_victims(self.cache.resident(), needed_tokens)
