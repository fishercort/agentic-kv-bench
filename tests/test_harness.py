"""Harness tests: caching semantics (compulsory vs capacity miss scoring),
LRU victim choice, and the load-bearing adoption proof — a 5-line custom policy
plugs in and runs."""

import pytest

from agentic_kv_bench.baselines import LRU
from agentic_kv_bench.harness import (
    BlockRef,
    CostParams,
    RequestAccess,
    replay,
)
from agentic_kv_bench.policy import Policy


def blk(bid, tokens=1, kind="history"):
    return BlockRef(block_id=bid, kind=kind, size_tokens=tokens)


def acc(ms, block_ids, tokens=1, events=None):
    return RequestAccess(
        arrival_ms=ms,
        blocks=[blk(b, tokens) for b in block_ids],
        lifecycle_events=events or [],
    )


COST = CostParams(recompute_ms_per_token=1.0)


def test_first_access_is_compulsory_not_scored():
    # 3 distinct blocks, huge cache: all compulsory, zero scored cost.
    trace = [acc(0, [1, 2, 3])]
    r = replay(trace, LRU(), COST, capacity_tokens=100)
    assert r.compulsory_misses == 3 and r.capacity_misses == 0
    assert r.scored_recompute_cost == 0.0


def test_resident_reaccess_is_a_hit():
    trace = [acc(0, [1, 2]), acc(1, [1, 2])]
    r = replay(trace, LRU(), COST, capacity_tokens=100)
    assert r.hits == 2 and r.capacity_misses == 0
    assert r.hit_rate == pytest.approx(2 / 4)


def test_capacity_miss_is_scored():
    # capacity 2 tokens. Access 1,2 (fills), then 3 (evicts LRU=1), then 1 again
    # (was evicted -> capacity miss, scored).
    trace = [acc(0, [1]), acc(1, [2]), acc(2, [3]), acc(3, [1])]
    r = replay(trace, LRU(), COST, capacity_tokens=2)
    assert r.capacity_misses == 1  # block 1 re-accessed after eviction
    assert r.scored_recompute_cost == 1.0  # 1 token * 1.0
    assert r.n_evictions >= 1


def test_lru_evicts_least_recently_used():
    # Isolate the victim choice: cache holds 2. Touch 1, then 2, then 1 again
    # (so 2 is LRU). Admit 3 -> LRU must evict 2, NOT 1. Then access 1 (hit,
    # proves 1 survived) and 2 (capacity miss, proves 2 was the victim).
    trace = [acc(0, [1]), acc(1, [2]), acc(2, [1]), acc(3, [3]), acc(4, [1]), acc(5, [2])]
    r = replay(trace, LRU(), COST, capacity_tokens=2)
    assert r.hits == 2  # acc2 [1] and acc4 [1]: 1 stayed resident
    assert r.capacity_misses == 1  # only 2 was evicted-then-reaccessed


def test_lru_thrashes_when_working_set_exceeds_capacity():
    # Capacity 2, cyclic working set of 3: LRU evicts exactly the block about to
    # be needed. Documents the thrash the benchmark exists to expose.
    trace = [acc(i, [(i % 3) + 1]) for i in range(6)]
    r = replay(trace, LRU(), COST, capacity_tokens=2)
    assert r.capacity_misses == 3  # every re-access after the first cycle misses


def test_high_value_hit_rate_tracks_expensive_blocks():
    # big block (10 tok) is expensive; capacity forces its eviction and return.
    trace = [acc(0, [1], tokens=10), acc(1, [2], tokens=10), acc(2, [1], tokens=10)]
    r = replay(trace, LRU(), COST, capacity_tokens=10, high_value_threshold=5.0)
    assert r.capacity_misses == 1 and r.high_value_capacity_misses == 1


def test_oversized_block_is_a_clean_error():
    trace = [acc(0, [1], tokens=50)]
    with pytest.raises(RuntimeError, match="exceeds"):
        replay(trace, LRU(), COST, capacity_tokens=10)


# -- the adoption proof: a custom policy plugs in with no framework changes ----


def test_custom_policy_plugs_in():
    """A researcher's policy is a subclass with one method. If this runs, the
    interface is a real contract, not a demo assumption."""

    class EvictBiggest(Policy):
        def evict(self, needed_tokens, now_ms):
            victims, freed = [], 0
            for m in sorted(
                self.cache.resident().values(),
                key=lambda m: m.size_tokens, reverse=True,
            ):
                victims.append(m.block_id)
                freed += m.size_tokens
                if freed >= needed_tokens:
                    break
            return victims

    trace = [acc(0, [1], 3), acc(1, [2], 1), acc(2, [3], 1), acc(3, [1], 3)]
    r = replay(trace, EvictBiggest(), COST, capacity_tokens=4)
    assert r.total_accesses == 4
    assert r.capacity_misses >= 1  # the big block got evicted and returned


def test_hints_reach_hint_consuming_policies():
    """A policy that acts on lifecycle hints receives them."""
    seen_events = []

    class HintWatcher(LRU):
        def on_hint(self, event, now_ms):
            seen_events.append(event["event"])

    trace = [
        acc(0, [1], events=[{"event": "compaction", "at_ms": 0}]),
        acc(1, [2]),
    ]
    replay(trace, HintWatcher(), COST, capacity_tokens=100)
    assert seen_events == ["compaction"]
