"""Oracle tests: cost-aware Belady correctness, the lower-bound property (no
online policy beats it under uniform cost), working-set protection, and
percent-of-oracle. Same rigor as the invariant checker: prove the yardstick is
correct before trusting what it measures."""

import random

import pytest

from agentic_kv_bench.baselines import LRU
from agentic_kv_bench.harness import BlockRef, CostParams, RequestAccess, replay
from agentic_kv_bench.oracle import oracle_run, percent_of_oracle
from agentic_kv_bench.policy import Policy

COST = CostParams(recompute_ms_per_token=1.0)


def blk(bid, tokens=1):
    return BlockRef(block_id=bid, kind="history", size_tokens=tokens)


def one(ms, bid):
    return RequestAccess(arrival_ms=ms, blocks=[blk(bid)])


def test_belady_evicts_farthest_future_use():
    # capacity 2. Accesses: 1,2,3,1,3,2. At the eviction (admitting 3 into {1,2}):
    # 1 is next used at step 3, 2 at step 5. Belady evicts 2 (farther). Then
    # 1 (hit), 3 (hit), 2 (capacity miss). LRU would evict 1 (older) and pay more.
    seq = [1, 2, 3, 1, 3, 2]
    trace = [one(i, b) for i, b in enumerate(seq)]
    ora = oracle_run(trace, COST, capacity_tokens=2)
    assert ora.capacity_misses == 1  # only block 2 recomputed


def test_oracle_is_a_lower_bound_under_uniform_cost():
    """Belady is provably optimal for uniform cost/size, so NO online policy
    can pay less than the oracle. Fuzz several policies, including a COMPETENT
    one (GDSF), and assert percent-of-oracle >= 100 every time. GDSF is the
    competent adversary the recorded note called for: a lower bound pressured
    only by LRU and random is validated against strawmen; GDSF is a real
    cost/frequency policy that could expose a gap in the bound if one existed."""
    from agentic_kv_bench.baselines import GDSF

    class Random(Policy):
        def __init__(self, seed):
            self.rng = random.Random(seed)

        def evict(self, needed_tokens, now_ms):
            ids = list(self.cache.resident())
            self.rng.shuffle(ids)
            victims, freed = [], 0
            for bid in ids:
                victims.append(bid)
                freed += self.cache.resident()[bid].size_tokens
                if freed >= needed_tokens:
                    break
            return victims

    for seed in range(20):
        rng = random.Random(seed)
        seq = [rng.randint(1, 6) for _ in range(40)]
        trace = [one(i, b) for i, b in enumerate(seq)]
        cap = 3
        ora = oracle_run(trace, COST, cap)
        for policy in (LRU(), Random(seed), GDSF()):
            res = replay(trace, policy, COST, cap)
            pct = percent_of_oracle(res, ora)
            assert pct >= 100.0 - 1e-9, f"policy beat the oracle: {pct} (seed {seed})"


def test_working_set_protection_oracle_and_harness():
    # A single request needs blocks {1,2,3} at once; capacity 3 fits exactly.
    # A later request needs {1,2,3,4}; capacity 3 cannot hold 4 at once -> error
    # on BOTH paths (same invariant).
    big = RequestAccess(arrival_ms=1, blocks=[blk(1), blk(2), blk(3), blk(4)])
    trace = [RequestAccess(arrival_ms=0, blocks=[blk(1), blk(2), blk(3)]), big]
    with pytest.raises(RuntimeError, match="working set exceeds capacity"):
        replay(trace, LRU(), COST, capacity_tokens=3)
    with pytest.raises(RuntimeError, match="working set exceeds capacity"):
        oracle_run(trace, COST, capacity_tokens=3)


def test_multi_block_request_does_not_evict_its_own_blocks():
    # request accesses {1,2,3} together; capacity 3. No block should be evicted
    # to admit its own sibling -> zero evictions, three compulsory misses.
    trace = [RequestAccess(arrival_ms=0, blocks=[blk(1), blk(2), blk(3)])]
    for run in (replay(trace, LRU(), COST, 3), oracle_run(trace, COST, 3)):
        assert run.n_evictions == 0 and run.compulsory_misses == 3


def test_percent_of_oracle_edges():
    from agentic_kv_bench.harness import RunResult

    z = RunResult(1, 1, 1, 0, 0.0, 0, 0, 1, 0)  # zero scored cost
    nz = RunResult(1, 0, 0, 1, 5.0, 5, 0, 1, 0)  # scored cost 5
    assert percent_of_oracle(z, z) == 100.0  # both zero -> matched
    assert percent_of_oracle(nz, z) == float("inf")  # policy pays, oracle didn't
    assert percent_of_oracle(nz, nz) == 100.0  # equal cost -> 100


def test_oracle_beats_lru_on_a_reuse_pattern():
    # A pattern where LRU thrashes but Belady keeps the reused block.
    seq = [1, 2, 1, 3, 1, 4, 1, 5]  # block 1 reused constantly, capacity 2
    trace = [one(i, b) for i, b in enumerate(seq)]
    ora = oracle_run(trace, COST, 2)
    lru = replay(trace, LRU(), COST, 2)
    assert ora.scored_recompute_cost <= lru.scored_recompute_cost
    assert percent_of_oracle(lru, ora) >= 100.0
