"""Bundled baseline policies, reimplemented from the literature per
docs/policy-interface.md. Fidelity matters (the adoption pitch to the papers'
authors is a faithful baseline), so each policy cites its source and rule, and
notes any simplification forced by the simulator's scope.

LRU is the floor. The ladder adds TTL (CacheTTL/Continuum), GDSF, WA-LRU
(SAGA), and retired-cache lifecycle.
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


class TTL(Policy):
    """Time-to-live eviction (CacheTTL / Continuum, arXiv 2511.02230).

    Mechanism: keep KV resident through tool-call gaps by giving each block a
    time-to-live; a block idle longer than its TTL is proactively expired to
    keep the cache lean for concurrent sessions. Under forced pressure it falls
    back to LRU. The paper derives TTL from reload cost and queueing delay;
    this reimplementation uses a fixed TTL parameter, because the adaptive
    derivation needs the live serving loop (queue state, reload timing) that
    the offline simulator does not model. Fidelity: the idle-time / gap-
    bridging mechanism is faithful; the adaptive-TTL derivation is simplified
    to a constant, noted here so the authors can see exactly what differs."""

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
