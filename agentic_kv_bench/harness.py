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

from dataclasses import dataclass, field

from agentic_kv_bench.policy import BlockMeta, CacheView, Policy


@dataclass(frozen=True)
class BlockRef:
    block_id: int
    kind: str
    size_tokens: int


@dataclass(frozen=True)
class RequestAccess:
    arrival_ms: int
    blocks: list[BlockRef]  # the prefix this request accesses, in order
    lifecycle_events: list[dict] = field(default_factory=list)


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


def replay(
    accesses: list[RequestAccess],
    policy: Policy,
    cost: CostParams,
    capacity_tokens: int,
    hints_enabled: bool = True,
    high_value_threshold: float | None = None,
) -> RunResult:
    resident: dict[int, BlockMeta] = {}
    view = CacheView(resident)
    policy.bind(view)
    seen: set[int] = set()

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
        if hints_enabled:
            for ev in req.lifecycle_events:
                policy.on_hint(ev, now)
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
