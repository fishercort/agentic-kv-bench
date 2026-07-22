"""The Port-driven simulator (scored).

`SimPort` is the reference Port and lives in `kv_policy_core` (the neutral package),
so an engine's conformance test can compare its own Port against the same reference
without depending on this benchmark. `replay_via_port` is this benchmark's scored
driver on top of that seam: same scoring as `harness.replay`, but every admission
and eviction goes through the Port. The parity harness (tests/test_parity.py)
asserts it reproduces the reference `replay` exactly; in miniserve, the same seam is
implemented by `MiniservePort` over the real block pool and checked against
`SimPort` there.
"""

from kv_policy_core import BlockMeta, SimPort

from agentic_kv_bench.harness import (
    _HINTS_ON,
    CostParams,
    HintDelivery,
    RequestAccess,
    RunResult,
    _build_hint_schedule,
)

__all__ = ["SimPort", "replay_via_port"]


def replay_via_port(
    accesses: list[RequestAccess],
    policy,
    cost: CostParams,
    capacity_tokens: int,
    hints: HintDelivery | None = None,
    high_value_threshold: float | None = None,
) -> RunResult:
    """The simulator driven through the Port seam. Scoring identical to
    harness.replay; the only difference is that admission/eviction go through
    SimPort, which is what miniserve swaps for MiniservePort."""
    hints = hints if hints is not None else _HINTS_ON
    port = SimPort(capacity_tokens)
    port.bind_policy(policy)
    seen: set = set()

    hits = compulsory = capacity = hv_capacity = 0
    scored_cost = 0.0
    scored_tokens = 0
    peak = 0
    schedule = _build_hint_schedule(accesses, hints)
    hp = 0

    for req in accesses:
        now = req.arrival_ms
        working = frozenset(b.block_id for b in req.blocks)
        port.set_protected(working)
        while hp < len(schedule) and schedule[hp][0] <= now:
            policy.on_hint(schedule[hp][1], schedule[hp][0])
            hp += 1
        for vid in policy.maintain(now):
            port.reap_if_resident(vid, working)
        for bref in req.blocks:
            m = port.get(bref.block_id)
            if m is not None:  # hit
                hits += 1
                m.last_access_ms = now
                m.access_count += 1
                policy.on_access(bref.block_id, m, now)
                continue
            rc = cost.recompute_cost(bref.size_tokens, bref.kind)
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
            port.free_to_fit(bref.size_tokens, now, working)
            meta = BlockMeta(
                block_id=bref.block_id, kind=bref.kind,
                size_tokens=bref.size_tokens, recompute_cost=rc,
                last_access_ms=now, access_count=1,
            )
            meta.tier = policy.place(meta)
            port.admit(meta)
            policy.on_access(bref.block_id, meta, now)
            peak = max(peak, port.resident_tokens)

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
        n_evictions=port.evictions,
    )
