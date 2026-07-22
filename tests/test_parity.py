"""Parity harness: the Port-driven simulator (replay_via_port over SimPort) must
reproduce the reference simulator (harness.replay) exactly, on every policy and
across the hint and per-kind-cost paths.

This is the step-1 form of the offline/online-parity check (docs/L2-integration-
design.md §8): one Port today (SimPort), so parity is measured against the
reference replay. Step 2 adds MiniservePort and this same harness compares the two
engines' decisions. If the seam is faithful, the two RunResults are identical.
"""

from agentic_kv_bench.baselines import (
    GDSF,
    LRU,
    WALRU,
    Continuum,
    EconomicJoint,
    GDSFHistory,
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
from agentic_kv_bench.ports import replay_via_port

UNIFORM = CostParams(recompute_ms_per_token=1.0)
PERKIND = CostParams(
    recompute_ms_per_token=1.0,
    kind_cost_multiplier={"reasoning": 1.0, "history": 1.0, "tool_output": 0.2},
)

# Fresh instance per run: policies are stateful, so parity must compare two
# independent instances driven down the two code paths.
POLICIES = {
    "LRU": lambda: LRU(),
    "GDSF": lambda: GDSF(),
    "GDSFHistory": lambda: GDSFHistory(),
    "WALRU_default": lambda: WALRU(),
    "WALRU_tuned": lambda: WALRU(alpha=1, beta=0.5, gamma=0),
    "IdleTTL": lambda: IdleTTL(ttl_ms=3),
    "RetiredCache": lambda: RetiredCache(),
    "Continuum": lambda: Continuum(gap_ms=20),
    "EconomicJoint": lambda: EconomicJoint(),
}


def blk(bid, tokens=1, kind="history"):
    return BlockRef(block_id=bid, kind=kind, size_tokens=tokens)


def acc(ms, blocks, events=None):
    return RequestAccess(arrival_ms=ms, blocks=blocks, lifecycle_events=events or [])


def _pressure_trace():
    # tuple block_ids (so Continuum's session extraction engages), cyclic working
    # set > capacity, a retire hint mid-stream, mixed kinds for pricing.
    kinds = ["history", "tool_output", "reasoning"]
    trace = []
    for i in range(24):
        sess = ("s", i % 3)
        b = (sess, i % 5)
        ev = None
        if i == 12:
            ev = [{"event": "retire", "at_ms": i, "block_ids": [(("s", 0), 0)]}]
        trace.append(acc(i, [blk(b, tokens=1, kind=kinds[i % 3])], events=ev))
    return trace


def _multiblock_trace():
    # multi-block requests + working-set protection + sizes.
    return [
        acc(0, [blk(1, 3), blk(2, 1)]),
        acc(1, [blk(3, 2), blk(1, 3)]),
        acc(2, [blk(4, 1), blk(2, 1), blk(5, 2)]),
        acc(3, [blk(1, 3)]),
        acc(4, [blk(3, 2), blk(4, 1)]),
    ]


def _run_parity(name, trace, cost, cap, hints=None, hv=None):
    ref = replay(trace, POLICIES[name](), cost, cap, hints=hints, high_value_threshold=hv)
    via = replay_via_port(trace, POLICIES[name](), cost, cap, hints=hints, high_value_threshold=hv)
    assert via == ref, (
        f"parity broken for {name}: port-driven != reference\n  ref={ref}\n  via={via}"
    )


def test_parity_all_policies_uniform_cost():
    trace = _pressure_trace()
    for name in POLICIES:
        _run_parity(name, trace, UNIFORM, cap=3)


def test_parity_all_policies_per_kind_cost():
    trace = _pressure_trace()
    for name in POLICIES:
        _run_parity(name, trace, PERKIND, cap=3)


def test_parity_multiblock_and_working_set():
    trace = _multiblock_trace()
    for name in POLICIES:
        _run_parity(name, trace, UNIFORM, cap=6)


def test_parity_under_hint_degradation():
    trace = _pressure_trace()
    for hints in (
        HintDelivery(enabled=False),
        HintDelivery(delay_ms=5),
        HintDelivery(drop_prob=0.5, seed=7),
    ):
        for name in ("RetiredCache", "Continuum", "EconomicJoint", "LRU"):
            _run_parity(name, trace, UNIFORM, cap=3, hints=hints)


def test_parity_high_value_threshold():
    trace = _multiblock_trace()
    for name in ("LRU", "GDSF", "EconomicJoint"):
        _run_parity(name, trace, UNIFORM, cap=6, hv=2.0)
