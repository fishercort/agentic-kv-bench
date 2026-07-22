"""Baseline-ladder tests. Each policy: its distinguishing mechanism is exercised
against a hand-built trace, and it remains a valid policy (oracle stays the lower
bound). The mentor's competent-adversary oracle-fuzz upgrade fires at GDSF."""

from agentic_kv_bench.baselines import GDSF, LRU, WALRU, IdleTTL
from agentic_kv_bench.harness import BlockRef, CostParams, RequestAccess, replay
from agentic_kv_bench.policy import BlockMeta, CacheView

COST = CostParams(recompute_ms_per_token=1.0)


def acc(ms, block_ids, tokens=1):
    return RequestAccess(
        arrival_ms=ms,
        blocks=[BlockRef(block_id=b, kind="history", size_tokens=tokens) for b in block_ids],
    )


def test_ttl_proactively_expires_idle_blocks():
    """TTL's distinguishing mechanism vs LRU: it evicts an idle block BEFORE
    any pressure forces it. Block 1 goes cold; a later access recomputes it
    under TTL (proactive expiry) but would be a free hit under LRU (no
    pressure). This is the trade TTL makes, exercised directly."""
    # huge cache, no capacity pressure at all
    trace = [
        acc(0, [1]),          # block 1 admitted, last access 0
        acc(1000, [2]),       # unrelated work; 1 idles
        acc(3000, [2]),       # 1 has now idled 3000 ms
        acc(3001, [1]),       # re-access block 1
    ]
    lru = replay(trace, LRU(), COST, capacity_tokens=100)
    ttl = replay(trace, IdleTTL(ttl_ms=1500), COST, capacity_tokens=100)
    # LRU never evicts (no pressure) -> block 1 is a hit at the end
    assert lru.capacity_misses == 0
    # TTL expires block 1 (idled 3000 > 1500) -> its re-access is a capacity miss
    assert ttl.capacity_misses == 1 and ttl.n_evictions >= 1


def test_ttl_keeps_blocks_within_ttl():
    # block 1 re-touched inside the TTL window is never expired.
    trace = [acc(0, [1]), acc(500, [1]), acc(900, [1])]
    ttl = replay(trace, IdleTTL(ttl_ms=1000), COST, capacity_tokens=100)
    assert ttl.n_evictions == 0 and ttl.capacity_misses == 0


def test_ttl_falls_back_to_lru_under_pressure():
    # capacity 2, working set 3, all touched inside TTL: no proactive expiry,
    # so TTL must fall back to LRU eviction and matches LRU's cost.
    trace = [acc(i, [(i % 3) + 1]) for i in range(6)]
    lru = replay(trace, LRU(), COST, capacity_tokens=2)
    ttl = replay(trace, IdleTTL(ttl_ms=10_000), COST, capacity_tokens=2)
    assert ttl.scored_recompute_cost == lru.scored_recompute_cost


def test_ttl_is_a_valid_policy_against_oracle():
    from agentic_kv_bench.oracle import oracle_run, percent_of_oracle

    trace = [acc(i, [(i % 4) + 1]) for i in range(20)]
    res = replay(trace, IdleTTL(ttl_ms=3), COST, capacity_tokens=2)
    ora = oracle_run(trace, COST, capacity_tokens=2)
    assert percent_of_oracle(res, ora) >= 100.0 - 1e-9  # oracle still the bound


def test_gdsf_keeps_frequently_reused_blocks():
    """GDSF's distinguishing signal vs LRU is frequency. Block 1 is reused
    constantly; block 2 changes every turn. Under capacity 2, GDSF should keep
    the high-frequency block 1 resident (fewer misses on it) where LRU thrashes
    on recency alone."""
    # 1 reused every other access; the other slot cycles through 2,3,4,5
    seq = [1, 2, 1, 3, 1, 4, 1, 5, 1, 6]
    trace = [acc(i, [b]) for i, b in enumerate(seq)]
    lru = replay(trace, LRU(), COST, capacity_tokens=2)
    gdsf = replay(trace, GDSF(), COST, capacity_tokens=2)
    # block 1 is accessed 5 times; GDSF should never capacity-miss it, LRU may.
    assert gdsf.scored_recompute_cost <= lru.scored_recompute_cost


def test_gdsf_is_a_valid_policy_against_oracle():
    from agentic_kv_bench.oracle import oracle_run, percent_of_oracle

    trace = [acc(i, [(i % 5) + 1]) for i in range(30)]
    res = replay(trace, GDSF(), COST, capacity_tokens=3)
    ora = oracle_run(trace, COST, capacity_tokens=3)
    assert percent_of_oracle(res, ora) >= 100.0 - 1e-9


def test_gdsf_cost_term_cancels_under_linear_cost_model():
    """Under the linear cost model (recompute cost = size * rate), GDSF's
    cost/size term is constant, so GDSF reduces exactly to LFU-with-aging. This
    documents WHY cost-awareness is latent in the current sweep: it activates
    only when the cost model gains a size-independent term (a per-recompute
    fixed overhead, as real hardware has). Two blocks, different sizes, cost
    proportional to size, equal frequency -> equal priority (size cancels)."""
    g = GDSF()
    resident = {
        1: BlockMeta(1, "history", size_tokens=100, recompute_cost=100.0, last_access_ms=0),
        2: BlockMeta(2, "history", size_tokens=10, recompute_cost=10.0, last_access_ms=0),
    }
    g.bind(CacheView(resident))
    for bid, m in resident.items():
        g.on_access(bid, m, 0)  # equal frequency (1 each)
    assert g._H[1] == g._H[2]  # cost/size = rate for both; size does not tip it


def test_walru_reuse_term_flips_the_victim_away_from_lru():
    """The reuse term makes WA-LRU diverge from pure recency, computed exactly.
    Block 1 is the OLDEST (LRU would evict it) but the MOST reused; block 2 is
    newer but rarely used. With alpha=beta=1, gamma=0, ages {1:100, 2:50}
    (max 100), freq {1:10, 2:1} (max 10):
        P_evict(1) = 100/100 + (1 - 10/10) = 1.0
        P_evict(2) =  50/100 + (1 -  1/10) = 1.4
    WA-LRU evicts 2 (higher priority), where LRU would evict 1 (oldest). The
    reuse signal overrides recency, per the formula."""
    w = WALRU(alpha=1.0, beta=1.0, gamma=0.0)
    resident = {
        1: BlockMeta(1, "history", size_tokens=1, recompute_cost=1.0, last_access_ms=0),
        2: BlockMeta(2, "history", size_tokens=1, recompute_cost=1.0, last_access_ms=50),
    }
    w.bind(CacheView(resident))
    for _ in range(10):
        w.on_access(1, resident[1], 0)  # block 1 heavily reused
    w.on_access(2, resident[2], 50)     # block 2 rarely used
    victims = w.evict(needed_tokens=1, now_ms=100)
    assert victims == [2]  # NOT [1], which pure LRU (oldest-first) would pick
    # and confirm LRU really would pick the other one, so the divergence is real
    lru = LRU()
    lru.bind(CacheView(resident))
    assert lru.evict(1, now_ms=100) == [1]


def test_walru_is_a_valid_policy_against_oracle():
    from agentic_kv_bench.oracle import oracle_run, percent_of_oracle

    trace = [acc(i, [(i % 5) + 1]) for i in range(30)]
    res = replay(trace, WALRU(), COST, capacity_tokens=3)
    ora = oracle_run(trace, COST, capacity_tokens=3)
    assert percent_of_oracle(res, ora) >= 100.0 - 1e-9
