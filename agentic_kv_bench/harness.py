"""The replay harness: run any policy against a trace under a memory budget,
scored against the cost model. The benchmark engine.

Caching semantics (the load-bearing correctness point): a block's FIRST access
is a compulsory miss (its prefill is unavoidable, every policy pays it, so it
is not scored). A later access to a block that was evicted is a CAPACITY miss:
its recompute cost IS scored, because that cost is the policy's eviction
decision made concrete. Only avoidable recompute counts. This is what makes
percent-of-oracle a fair comparison.

Cost is consumed structurally per the Phase 2 verdict: recompute cost per token
is a parameter (the v1 crossover was degenerate), swept by the caller.
"""

import random
from dataclasses import dataclass, field

from agentic_kv_bench.policy import BlockMeta, CacheView, Policy


@dataclass(frozen=True)
class HintDelivery:
    """How lifecycle hints reach the policy this run: the degradation switches
    (docs/hint-interface.md) that make the hint interface a robustness CURVE
    rather than an on/off demo. The four positions the spec names:

        on       HintDelivery()                     enabled, no delay, no drop
        off      HintDelivery(enabled=False)        policy sees no hints
        delayed  HintDelivery(delay_ms=N)           each hint arrives N ms late
        dropped  HintDelivery(drop_prob=p, seed=s)  each hint lost w.p. p

    Delay and drop compose (a channel can be both late and lossy). Drop is
    seeded so a lossy run is reproducible, same discipline as the oracle fuzz."""

    enabled: bool = True
    delay_ms: int = 0
    drop_prob: float = 0.0
    seed: int = 0


_HINTS_ON = HintDelivery()


@dataclass(frozen=True)
class BlockRef:
    block_id: object  # any hashable, opaque to the harness; sessions namespace it
    kind: str
    size_tokens: int


@dataclass(frozen=True)
class RequestAccess:
    arrival_ms: int
    blocks: list[BlockRef]  # the prefix this request accesses, in order
    lifecycle_events: list[dict] = field(default_factory=list)


def interleave(
    sessions: list[list[RequestAccess]], gap_ms: int = 1000
) -> list[RequestAccess]:
    """Stitch multiple sessions onto one timeline so they compete for the cache.

    A single conversation never evicts (its own prefix fits); the benchmark's
    memory pressure is many concurrent sessions. v1 overlay: session k starts at
    k * gap_ms and its internal arrival times shift by that offset. This is the
    deterministic stand-in for the Poisson / Azure-calibrated arrival process
    that trace-schema.md specifies as the calibrated refinement. Block ids must
    already be namespaced per session (see access.access_from_source).
    """
    merged: list[RequestAccess] = []
    for k, session in enumerate(sessions):
        offset = k * gap_ms
        for req in session:
            merged.append(
                RequestAccess(
                    arrival_ms=req.arrival_ms + offset,
                    blocks=req.blocks,
                    lifecycle_events=req.lifecycle_events,
                )
            )
    merged.sort(key=lambda r: r.arrival_ms)
    return merged


@dataclass
class CostParams:
    """Structural cost model (Phase 2 verdict): recompute is a swept parameter,
    not the degenerate v1 constant. Migration priced per tier for offloading
    policies."""

    recompute_ms_per_token: float = 1.0
    migrate_ms_per_token: dict[str, float] = field(
        default_factory=lambda: {"cpu": 0.05, "disk": 0.5}
    )

    def recompute_cost(self, size_tokens: int) -> float:
        return size_tokens * self.recompute_ms_per_token


@dataclass
class RunResult:
    total_accesses: int
    hits: int
    compulsory_misses: int  # first prefill, unavoidable, not scored
    capacity_misses: int  # evicted-then-reaccessed, scored
    scored_recompute_cost: float  # the headline: avoidable recompute
    scored_recompute_tokens: int
    high_value_capacity_misses: int  # capacity misses on expensive-to-recompute blocks
    peak_resident_tokens: int
    n_evictions: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_accesses if self.total_accesses else 0.0


def _build_hint_schedule(
    accesses: list[RequestAccess], hints: HintDelivery
) -> list[tuple[int, dict]]:
    """Flatten every request's lifecycle events into a delivery-time-ordered
    schedule, applying the degradation switches: drop with probability p
    (seeded), shift delivery by delay_ms. An event's delivery time is its own
    at_ms (when it happened) plus the delay; the harness delivers it the first
    time the replay clock reaches that time (see replay). Off -> empty schedule."""
    if not hints.enabled:
        return []
    rng = random.Random(hints.seed)
    schedule: list[tuple[int, dict]] = []
    for req in accesses:
        for ev in req.lifecycle_events:
            if hints.drop_prob and rng.random() < hints.drop_prob:
                continue  # hint lost in transit
            at = ev.get("at_ms", req.arrival_ms)
            schedule.append((at + hints.delay_ms, ev))
    schedule.sort(key=lambda x: x[0])  # stable: ties keep emission order
    return schedule


def replay(
    accesses: list[RequestAccess],
    policy: Policy,
    cost: CostParams,
    capacity_tokens: int,
    hints: HintDelivery | None = None,
    high_value_threshold: float | None = None,
) -> RunResult:
    hints = hints if hints is not None else _HINTS_ON
    resident: dict[int, BlockMeta] = {}
    view = CacheView(resident)
    policy.bind(view)
    seen: set[int] = set()
    schedule = _build_hint_schedule(accesses, hints)
    hp = 0  # pointer into schedule; hints deliver as the clock passes them

    hits = compulsory = capacity = evictions = hv_capacity = 0
    scored_cost = 0.0
    scored_tokens = 0
    peak = 0
    resident_tokens = 0

    def free_to_fit(need: int, now: int, protected: frozenset[int]) -> None:
        nonlocal resident_tokens, evictions
        while capacity_tokens - resident_tokens < need:
            evictable = sum(
                m.size_tokens for b, m in resident.items() if b not in protected
            )
            if evictable < need - (capacity_tokens - resident_tokens):
                raise RuntimeError(
                    "request working set exceeds capacity: its prefix cannot be "
                    "held resident even after evicting everything evictable"
                )
            victims = policy.evict(need - (capacity_tokens - resident_tokens), now)
            if not victims:
                raise RuntimeError(
                    "policy.evict returned no victims but space is needed; "
                    "a correct policy must free space or the request cannot be admitted"
                )
            for vid in victims:
                if vid in protected or vid not in resident:
                    raise RuntimeError(
                        f"policy tried to evict block {vid}, which is "
                        f"{'needed by the current request' if vid in protected else 'not resident'}"
                    )
                m = resident.pop(vid)
                resident_tokens -= m.size_tokens
                evictions += 1

    for req in accesses:
        now = req.arrival_ms
        working = frozenset(b.block_id for b in req.blocks)
        view._set_protected(working)
        # Deliver every hint whose (possibly delayed) delivery time has arrived,
        # BEFORE maintain(), so a lifecycle policy can reclaim freshly-retired
        # blocks in the same step. now_ms is the delivery time (when the policy
        # observes it), which may be earlier than this request under delay=0.
        while hp < len(schedule) and schedule[hp][0] <= now:
            policy.on_hint(schedule[hp][1], schedule[hp][0])
            hp += 1
        # Proactive expiry (TTL-style / lifecycle reclamation), before admission.
        # Never touches the current working set or a non-resident id.
        for vid in policy.maintain(now):
            if vid in resident and vid not in working:
                m = resident.pop(vid)
                resident_tokens -= m.size_tokens
                evictions += 1
        for bref in req.blocks:
            m = resident.get(bref.block_id)
            if m is not None:  # hit
                hits += 1
                m.last_access_ms = now
                m.access_count += 1
                policy.on_access(bref.block_id, m, now)
                continue
            # miss: admit the block, evicting if needed.
            rc = cost.recompute_cost(bref.size_tokens)
            first_time = bref.block_id not in seen
            if first_time:
                compulsory += 1
                seen.add(bref.block_id)
            else:
                capacity += 1
                scored_cost += rc
                scored_tokens += bref.size_tokens
                if high_value_threshold is not None and rc >= high_value_threshold:
                    hv_capacity += 1
            if bref.size_tokens > capacity_tokens:
                raise RuntimeError(
                    f"block {bref.block_id} ({bref.size_tokens} tok) exceeds "
                    f"capacity ({capacity_tokens} tok); no policy can admit it"
                )
            free_to_fit(bref.size_tokens, now, working)
            meta = BlockMeta(
                block_id=bref.block_id, kind=bref.kind,
                size_tokens=bref.size_tokens, recompute_cost=rc,
                last_access_ms=now, access_count=1,
            )
            meta.tier = policy.place(meta)
            resident[bref.block_id] = meta
            resident_tokens += bref.size_tokens
            policy.on_access(bref.block_id, meta, now)
            peak = max(peak, resident_tokens)

    # Drain hints whose delivery time falls after the last request. No accesses
    # follow, so they cannot change scoring; delivered for contract completeness
    # (a delayed hint is still observed, just too late to matter).
    while hp < len(schedule):
        policy.on_hint(schedule[hp][1], schedule[hp][0])
        hp += 1

    return RunResult(
        total_accesses=hits + compulsory + capacity,
        hits=hits,
        compulsory_misses=compulsory,
        capacity_misses=capacity,
        scored_recompute_cost=scored_cost,
        scored_recompute_tokens=scored_tokens,
        high_value_capacity_misses=hv_capacity,
        peak_resident_tokens=peak,
        n_evictions=evictions,
    )
