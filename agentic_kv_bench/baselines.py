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


class WALRU(Policy):
    """Workflow-Aware LRU (SAGA, arXiv 2605.00528).

    Exact scoring rule (paper's Eq., eviction priority; evict the highest):

        P_evict(b) = alpha * R_hat(b) + beta * (1 - P_reuse(b)) + gamma * S_hat(b)

    where R_hat is normalized recency (age since last access), P_reuse is the
    block's reuse probability, and S_hat is normalized size. A block that is
    stale, unlikely to be reused, and large is the first to go.

    Fidelity, at its highest bar because the authors may check, with two
    disclosed gaps so they see exactly what differs:
    1. P_reuse SOURCE: the paper derives it from the Agent Execution Graph
       (workflow structure). This is the INFERENCE-ONLY version, estimating
       P_reuse from normalized access frequency (no graph, no hints). The
       faithful graph-driven WA-LRU is the hint-consuming rung and lands with
       the hint interface; this row is its signal-blind lower bound.
    2. WEIGHTS: alpha/beta/gamma are set to neutral defaults (1, 1, 0), not the
       paper's tuned values (not extracted here); the sweep varies them. gamma
       defaults to 0 because S_hat is vestigial under uniform block size (same
       cancellation as GDSF's cost term).
    The three-term RULE itself is faithful; the estimator and weights are the
    named substitutions."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.0):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self._freq: dict = {}

    def on_access(self, block_id, meta, now_ms: int) -> None:
        # P_reuse estimator: accumulated access frequency (retained across
        # residency gaps, since P_reuse is meant to be a stable estimate, not a
        # per-residency count; contrast GDSF's canonical reset).
        self._freq[block_id] = self._freq.get(block_id, 0) + 1

    def evict(self, needed_tokens: int, now_ms: int) -> list:
        resident = self.cache.resident()
        ages = {b: now_ms - m.last_access_ms for b, m in resident.items()}
        max_age = max(ages.values()) or 1
        max_freq = max((self._freq.get(b, 1) for b in resident), default=1) or 1
        max_size = max((m.size_tokens for m in resident.values()), default=1) or 1

        def p_evict(b: object) -> float:
            r_hat = ages[b] / max_age
            p_reuse = self._freq.get(b, 1) / max_freq
            s_hat = resident[b].size_tokens / max_size
            return self.alpha * r_hat + self.beta * (1 - p_reuse) + self.gamma * s_hat

        by_priority = sorted(resident, key=p_evict, reverse=True)  # highest first
        victims, freed = [], 0
        for b in by_priority:
            victims.append(b)
            freed += resident[b].size_tokens
            if freed >= needed_tokens:
                break
        return victims


class RetiredCache(Policy):
    """Retired-cache lifecycle eviction (arXiv 2605.06472). The designated
    hint-interface consumer: workflow lifecycle signals mark blocks DEAD, and
    the policy reclaims them proactively instead of waiting for recency to age
    them out. On this corpus the lifecycle signal is the compaction retirement
    hint (the framework compacted the prefix, so it knows exactly which KV it
    dropped); in the paper it is the termination message. Same shape: the
    message IS the hint.

    This is the rung the whole ladder was built toward, because it is the first
    policy that can see what recency and frequency provably cannot - which
    blocks are dead vs merely old (the 'idle != dead' gap the IdleTTL and GDSF
    rungs both ran into). Its value is exactly the retirement signal, so it is
    the natural place to measure with-hints vs inference-only.

    Graceful degradation, the contract that makes this a real interface: with
    hints OFF (or fully dropped) no block is ever marked retired, so maintain()
    is empty and evict() is pure LRU. RetiredCache-inference-only IS LRU, by
    construction - the policy adds signal, never subtracts it. The with-hints
    vs hints-off gap is therefore precisely what the retirement signal buys."""

    def __init__(self):
        self._retired: set = set()

    def on_hint(self, event: dict, now_ms: int) -> None:
        if event.get("event") == "retire":
            self._retired.update(event.get("block_ids", ()))

    def maintain(self, now_ms: int) -> list:
        # Proactively reclaim retired blocks still resident. resident() already
        # excludes the current working set, so a block needed now is never here.
        return [b for b in self.cache.resident() if b in self._retired]

    def evict(self, needed_tokens: int, now_ms: int) -> list:
        resident = self.cache.resident()
        victims, freed = [], 0
        # Retired (dead) blocks first: evicting them is free, they never return.
        for b in resident:
            if b in self._retired:
                victims.append(b)
                freed += resident[b].size_tokens
                if freed >= needed_tokens:
                    return victims
        # Then LRU among the living. (With hints off, _retired is empty and this
        # is the only branch, so the policy is exactly LRU.)
        for m in sorted(resident.values(), key=lambda m: m.last_access_ms):
            if m.block_id in self._retired:
                continue
            victims.append(m.block_id)
            freed += m.size_tokens
            if freed >= needed_tokens:
                break
        return victims


class EconomicJoint(Policy):
    """The contender: the economic joint policy (docs/policy-interface.md). Every
    published baseline answers 'which block do I drop?'; this one prices the drop.
    It scores each resident block by the expected recompute cost of losing it and
    evicts the cheapest-to-lose first:

        price(b) = 0                              if b is retired (dead, P_reuse=0)
                 = recompute_cost(b) / (age + 1)  otherwise

    where age = now - last_access. This is the JOINT policy: it consults all
    three signals at once - lifecycle (retired => price 0, evict dead first),
    cost (recompute_cost, so expensive-to-rebuild blocks are protected), and
    reuse (recency via 1/age, the estimator the WA-LRU rung showed is the useful
    subordinate correction). No single baseline uses all three.

    Parity by construction: under UNIFORM cost every block's recompute_cost is
    identical, so price collapses to a constant/age and the ranking is pure
    recency, with retired blocks first - exactly RetiredCache (dead-first + LRU).
    The uniform-cost panel is therefore a parity CHECK on the implementation: if
    it does not reproduce RetiredCache's curve, the lifecycle/recency wiring is
    wrong. Only under per-kind (heterogeneous) cost does the pricing term move
    the ranking, which is the cost-swept panel where pricing earns its keep or
    does not.

    Scope: the paper's four-way decision is evict / offload / recompute-later /
    refuse. Offload and refuse need measured migrate costs and tier-aware scoring
    (Phase 4); this rung prices the eviction decision only and defers the other
    two, stated not hidden. With hints off it degrades to cost-weighted recency
    (and to plain LRU under uniform cost), never below the floor by construction."""

    def __init__(self):
        self._retired: set = set()

    def on_hint(self, event: dict, now_ms: int) -> None:
        if event.get("event") == "retire":
            self._retired.update(event.get("block_ids", ()))

    def maintain(self, now_ms: int) -> list:
        return [b for b in self.cache.resident() if b in self._retired]

    def evict(self, needed_tokens: int, now_ms: int) -> list:
        resident = self.cache.resident()

        def price(b: object):
            m = resident[b]
            if b in self._retired:
                return (0.0, m.last_access_ms)  # dead: free to lose, evict first
            age = now_ms - m.last_access_ms + 1
            return (m.recompute_cost / age, m.last_access_ms)

        order = sorted(resident, key=price)  # cheapest-to-lose first
        victims, freed = [], 0
        for b in order:
            victims.append(b)
            freed += resident[b].size_tokens
            if freed >= needed_tokens:
                break
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
