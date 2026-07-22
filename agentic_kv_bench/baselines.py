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


class GDSF(Policy):
    """GreedyDual-Size-Frequency (Cherkasova 1998, extending Cao & Irani's
    GreedyDual-Size, 1997).

    Priority H(b) = L + freq(b) * cost(b) / size(b), where L is an aging clock.
    Evict the minimum-H block and advance L to its H, so freshly admitted or
    frequently reused or expensive-to-rebuild blocks are retained. This is the
    first COST-CONSULTING policy: value scales with recompute cost.

    Cost-awareness is LATENT under the current linear cost model (cost = size *
    rate): cost/size cancels to a constant, so GDSF here IS GreedyDual-Frequency
    (LFU with aging). It differentiates on cost only when the cost model gains a
    size-independent term (a per-recompute fixed overhead, as real hardware has).

    Frequency-reset fork, decided not inherited: this is the CANONICAL Cherkasova
    version, where frequency is per-residency (evict() drops the victim's freq,
    so a re-fetched block restarts at freq 1). The retained-history variant
    (frequency survives residency gaps) is the known improvement and is run as an
    ablation, not silently shipped. On this workload the canonical reset risks a
    self-reinforcing thrash loop: agentic re-reads bring an evicted hot block
    straight back as a freq-1 stranger, making it the next cheapest victim. If
    that loop appears in the ladder, it is a finding with a mechanism, and the
    history variant becomes a one-line ablation with a story."""

    def __init__(self):
        self._L = 0.0
        self._H: dict = {}
        self._freq: dict = {}

    def on_access(self, block_id, meta, now_ms: int) -> None:
        self._freq[block_id] = self._freq.get(block_id, 0) + 1
        self._H[block_id] = self._L + (
            self._freq[block_id] * meta.recompute_cost / meta.size_tokens
        )

    def evict(self, needed_tokens: int, now_ms: int) -> list:
        resident = self.cache.resident()
        by_h = sorted(resident, key=lambda b: self._H.get(b, self._L))
        victims, freed = [], 0
        for bid in by_h:
            # Batch-eviction generalization of GreedyDual: the original evicts
            # singly and advances L to that victim's H; here L ratchets to each
            # victim's H in ascending order as we free multiple blocks.
            self._L = self._H.get(bid, self._L)
            victims.append(bid)
            freed += resident[bid].size_tokens
            if freed >= needed_tokens:
                break
        for bid in victims:  # evicted blocks leave the priority bookkeeping
            self._H.pop(bid, None)
            self._freq.pop(bid, None)  # canonical: frequency is per-residency
        return victims


class GDSFHistory(GDSF):
    """Retained-history GDSF ablation: frequency survives residency gaps, so an
    evicted-then-refetched block re-enters with its accumulated standing instead
    of as a freq-1 stranger. The known improvement over canonical GDSF; run to
    test whether the canonical reset-thrash mechanism (see GDSF docstring) is
    what makes plain GDSF underperform LRU on agentic re-reads."""

    def evict(self, needed_tokens: int, now_ms: int) -> list:
        resident = self.cache.resident()
        by_h = sorted(resident, key=lambda b: self._H.get(b, self._L))
        victims, freed = [], 0
        for bid in by_h:
            self._L = self._H.get(bid, self._L)
            victims.append(bid)
            freed += resident[bid].size_tokens
            if freed >= needed_tokens:
                break
        for bid in victims:
            self._H.pop(bid, None)  # H is rebuilt on next access; freq RETAINED
        return victims


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
