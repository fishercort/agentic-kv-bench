"""Bundled baseline policies. LRU is the floor; more (CacheTTL, GDSF, WA-LRU,
retired-cache) reimplement the literature per docs/policy-interface.md.
"""

from agentic_kv_bench.policy import Policy


class LRU(Policy):
    """Evict least-recently-accessed blocks. The floor: no cost awareness, no
    lifecycle awareness. Reads recency straight off the resident view, so it
    holds zero shadow state."""

    def evict(self, needed_tokens: int, now_ms: int) -> list[int]:
        resident = self.cache.resident()
        by_recency = sorted(resident.values(), key=lambda m: m.last_access_ms)
        victims, freed = [], 0
        for m in by_recency:
            victims.append(m.block_id)
            freed += m.size_tokens
            if freed >= needed_tokens:
                break
        return victims
