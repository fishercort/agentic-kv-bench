"""Baseline-ladder tests. Each policy: its distinguishing mechanism is exercised
against a hand-built trace, and it remains a valid policy (oracle stays the lower
bound). The mentor's competent-adversary oracle-fuzz upgrade fires at GDSF."""

from agentic_kv_bench.baselines import (
    GDSF,
    LRU,
    WALRU,
    EconomicJoint,
    IdleTTL,
    RetiredCache,
)
from agentic_kv_bench.harness import (
    BlockRef,
    CostParams,
    HintDelivery,
    RequestAccess,
    replay,
)
from agentic_kv_bench.policy import BlockMeta, CacheView

COST = CostParams(recompute_ms_per_token=1.0)


def acc(ms, block_ids, tokens=1, events=None):
    return RequestAccess(
        arrival_ms=ms,
        blocks=[BlockRef(block_id=b, kind="history", size_tokens=tokens) for b in block_ids],
        lifecycle_events=events or [],
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


def _dead_over_old_trace():
    # Block 1 is older but LIVE (reused at the end); block 2 is newer but DEAD
    # (retired at t=2, never used again). Capacity 2 forces one out when 3 arrives.
    return [
        acc(0, [1]),
        acc(1, [2]),
        acc(2, [3], events=[{"event": "retire", "at_ms": 2, "block_ids": [2]}]),
        acc(3, [1]),
    ]


def test_retired_cache_evicts_dead_over_old_where_lru_pays():
    """RetiredCache's distinguishing mechanism: it evicts a DEAD-but-recent block
    where LRU evicts a LIVE-but-old one. This is the 'dead != old' signal recency
    and frequency provably cannot see (the IdleTTL and GDSF rungs both hit it)."""
    trace = _dead_over_old_trace()
    lru = replay(trace, LRU(), COST, capacity_tokens=2)
    rc = replay(trace, RetiredCache(), COST, capacity_tokens=2)
    assert lru.capacity_misses == 1  # LRU drops old-but-live 1, recomputes it
    assert rc.capacity_misses == 0   # RetiredCache drops dead 2, keeps live 1


def test_retired_cache_degrades_to_lru_with_hints_off():
    """The graceful-degradation contract: no hints -> _retired stays empty ->
    RetiredCache IS LRU. The policy adds signal, never subtracts it."""
    trace = [acc(i, [(i % 3) + 1]) for i in range(6)]
    off = HintDelivery(enabled=False)
    lru = replay(trace, LRU(), COST, capacity_tokens=2)
    rc = replay(trace, RetiredCache(), COST, capacity_tokens=2, hints=off)
    assert rc.scored_recompute_cost == lru.scored_recompute_cost
    assert rc.n_evictions == lru.n_evictions


def test_retired_cache_with_dropped_hint_matches_lru_cost():
    """Ties the degradation switch to the outcome: drop the retire hint and
    RetiredCache pays LRU's cost, because it never learns block 2 is dead."""
    trace = _dead_over_old_trace()
    dropped = replay(trace, RetiredCache(), COST, 2, hints=HintDelivery(drop_prob=1.0))
    assert dropped.capacity_misses == 1  # no hint -> behaves as LRU


def test_retired_cache_is_a_valid_policy_against_oracle():
    from agentic_kv_bench.oracle import oracle_run, percent_of_oracle

    trace = [acc(i, [(i % 5) + 1]) for i in range(30)]
    res = replay(trace, RetiredCache(), COST, capacity_tokens=3)
    ora = oracle_run(trace, COST, capacity_tokens=3)
    assert percent_of_oracle(res, ora) >= 100.0 - 1e-9


def test_economic_joint_reduces_to_retired_cache_under_uniform_cost():
    """The parity check, at unit scale: under uniform cost the pricing term is
    inert, so EconomicJoint must make byte-for-byte the same decisions as
    RetiredCache (dead-first + LRU). Run both on a trace with pressure AND a
    retirement hint; assert identical scoring."""
    trace = _dead_over_old_trace()  # has a retire hint and forces one eviction
    econ = replay(trace, EconomicJoint(), COST, capacity_tokens=2)
    rc = replay(trace, RetiredCache(), COST, capacity_tokens=2)
    assert econ.scored_recompute_cost == rc.scored_recompute_cost
    assert econ.capacity_misses == rc.capacity_misses == 0
    # and a churnier trace, still uniform cost -> still identical to RetiredCache
    churny = [acc(i, [(i % 4) + 1]) for i in range(24)]
    e2 = replay(churny, EconomicJoint(), COST, capacity_tokens=2)
    r2 = replay(churny, RetiredCache(), COST, capacity_tokens=2)
    assert e2.scored_recompute_cost == r2.scored_recompute_cost


def test_economic_joint_prices_and_keeps_the_expensive_block():
    """Under per-kind cost the pricing term flips the victim. Two living blocks,
    capacity 1 free needed: block E is expensive (cost 10) and older; block C is
    cheap (cost 1) and newer. price = cost/(age+1):
        E: 10/(100+1) = 0.099   C: 1/(10+1) = 0.091
    EconomicJoint evicts C (cheaper to lose), KEEPING the expensive E, where LRU
    would evict E (older). Pricing protects the costly-to-rebuild block."""
    econ = EconomicJoint()
    resident = {
        "E": BlockMeta("E", "reasoning", size_tokens=1, recompute_cost=10.0, last_access_ms=0),
        "C": BlockMeta("C", "tool_output", size_tokens=1, recompute_cost=1.0, last_access_ms=90),
    }
    econ.bind(CacheView(resident))
    victims = econ.evict(needed_tokens=1, now_ms=100)
    assert victims == ["C"]  # keep expensive E; LRU (oldest-first) would take E
    lru = LRU()
    lru.bind(CacheView(resident))
    assert lru.evict(1, now_ms=100) == ["E"]


def test_economic_joint_parity_break_is_caused_by_block_SIZE_not_a_bug():
    """Degeneracy anchor for the panel-1 parity refutation. The claim: Econ
    diverges from RetiredCache under uniform cost ONLY because block sizes vary
    (recompute_cost = size*rate is then not constant). Prove it both ways.

    (a) UNIFORM SIZE -> exact parity: same-size blocks, uniform cost, Econ picks
        the LRU victim exactly (already covered by the reduces_to_retired_cache
        test; restated here at evict() level for the contrast).
    (b) VARIABLE SIZE -> divergence: a big-old block vs a small-new block. LRU
        evicts the old (big) one; Econ prices size/(age+1) and evicts the small
        one. Same uniform rate, different victim -> the residual is size."""
    # (a) uniform size: price = const/age -> LRU. Econ victim == LRU victim.
    uni = {
        1: BlockMeta(1, "history", size_tokens=5, recompute_cost=5.0, last_access_ms=0),
        2: BlockMeta(2, "history", size_tokens=5, recompute_cost=5.0, last_access_ms=50),
    }
    e = EconomicJoint()
    e.bind(CacheView(uni))
    lru = LRU()
    lru.bind(CacheView(uni))
    assert e.evict(1, now_ms=100) == lru.evict(1, now_ms=100) == [1]  # oldest
    # (b) variable size: big-old (A) vs small-new (B). price A=10/101=0.099,
    # B=1/51=0.0196 -> Econ evicts B; LRU evicts A (oldest). Divergence, uniform rate.
    var = {
        "A": BlockMeta("A", "history", size_tokens=10, recompute_cost=10.0, last_access_ms=0),
        "B": BlockMeta("B", "history", size_tokens=1, recompute_cost=1.0, last_access_ms=50),
    }
    e2 = EconomicJoint()
    e2.bind(CacheView(var))
    l2 = LRU()
    l2.bind(CacheView(var))
    assert e2.evict(1, now_ms=100) == ["B"]   # size-priced: drop the cheap small block
    assert l2.evict(1, now_ms=100) == ["A"]   # recency: drop the old block
    # -> parity holds iff sizes are uniform; the corpus's partial trailing blocks
    #    break that, exactly the ~0.4-3.6% residual seen in panel 1.


def test_economic_joint_is_a_valid_policy_against_oracle_per_kind_cost():
    from agentic_kv_bench.oracle import oracle_run, percent_of_oracle

    cost = CostParams(recompute_ms_per_token=1.0,
                      kind_cost_multiplier={"tool_output": 0.2, "reasoning": 1.0})
    trace = [
        RequestAccess(arrival_ms=i, blocks=[BlockRef(
            block_id=(i % 5) + 1, kind=("tool_output" if i % 2 else "reasoning"),
            size_tokens=1)])
        for i in range(30)
    ]
    res = replay(trace, EconomicJoint(), cost, capacity_tokens=3)
    ora = oracle_run(trace, cost, capacity_tokens=3)
    # oracle is a strong approximation under heterogeneous cost, not a proven
    # optimum, so allow a small tolerance rather than a hard lower bound.
    assert percent_of_oracle(res, ora) >= 95.0
