"""Locally-testable parts of the box-only runtime glue: the minute-one sanity checks
(subscribe) and the sharding/pacing (drive). The network paths (zmq, httpx) are exercised
on the box; their inputs are decided here."""

import pytest

from agentic_kv_bench.vllm_leg.drive import (
    build_workload,
    clamp_prompt,
    pace,
    select_by_footprint,
    shard_traces,
    trace_footprint,
)
from agentic_kv_bench.vllm_leg.kv_events import SchemaDriftError
from agentic_kv_bench.vllm_leg.mock_stream import GOLDEN, GOLDEN_FINGERPRINT
from agentic_kv_bench.vllm_leg.subscribe import check_schema, seed_agreement, seq_gap


# --- minute-one sanity checks -------------------------------------------------
def test_check_schema_accepts_golden_and_flags_unknown_shape():
    good = GOLDEN["streams"][0][0]
    fp, is_known = check_schema(good, GOLDEN_FINGERPRINT[0])
    assert is_known
    # a shape not in the pinned golden decodes fine but is flagged as a logged diff
    novel = [9.0, [["BlockRemoved", [1, 2, 3], "GPU"]], 0]
    fp2, is_known2 = check_schema(novel, GOLDEN_FINGERPRINT[0])
    assert not is_known2


def test_check_schema_raises_on_real_drift():
    with pytest.raises(SchemaDriftError):
        check_schema([1.0, [["BlockTeleported", [1], "GPU"]], 0])


def test_seq_gap_detects_dropped_messages():
    assert seq_gap(None, 5) == 0     # first message, no baseline
    assert seq_gap(5, 6) == 0        # in order
    assert seq_gap(5, 9) == 3        # dropped 6,7,8
    assert seq_gap(5, 5) == 0        # duplicate / no advance
    assert seq_gap(9, 2) == 0        # publisher restart (reset), not a drop


def test_seed_agreement_detects_pythonhashseed_mismatch():
    # identical probe prefix -> hashes must match across instances (shared NONE_HASH)
    assert seed_agreement({0: [b"h1", b"h2", b"h3"], 1: [b"h1", b"h2", b"h3"]})
    # different NONE_HASH -> prefix-chained hashes diverge -> overlap silently zero
    assert not seed_agreement({0: [b"h1", b"h2"], 1: [b"xx", b"yy"]})
    assert seed_agreement({0: [b"h1"]})  # single instance trivially agrees


# --- sharding / pacing --------------------------------------------------------
def _traces(n):
    return [{"id": f"trace_{i:04d}", "models": ["opus"],
             "requests": [{"t": 0.0, "out": 5, "hash_ids": [1, 2, 3]}]} for i in range(n)]


def test_shard_is_deterministic_and_total_preserving():
    tr = _traces(20)
    a = shard_traces(tr, 2)
    b = shard_traces(list(reversed(tr)), 2)
    assert a == b  # order-independent
    assert sum(len(v) for v in a.values()) == 20  # no trace dropped or duplicated
    ids = [t["id"] for v in a.values() for t in v]
    assert len(set(ids)) == 20


def test_pace_deltas_follow_arrival_times():
    reqs = [{"t": 0.0}, {"t": 2.0}, {"t": 2.5}]
    delays = [d for d, _ in pace(reqs)]
    assert delays == [0.0, 2.0, 0.5]
    # asap collapses all delays
    assert [d for d, _ in pace(reqs, speedup=float("inf"))] == [0.0, 0.0, 0.0]
    # speedup halves wall-clock
    assert [d for d, _ in pace(reqs, speedup=2.0)] == [0.0, 1.0, 0.25]


def test_clamp_prompt_keeps_prompt_plus_output_under_context():
    # a prompt that would overflow max_model_len is truncated to the prefix (no 400)
    toks = list(range(1000))
    out = clamp_prompt(toks, max_tokens=5, max_model_len=500)
    assert len(out) == 495 and out == toks[:495]      # prompt+output == max_model_len
    # a prompt that already fits is untouched (prefix, not a copy-mangle)
    small = list(range(100))
    assert clamp_prompt(small, max_tokens=1, max_model_len=500) == small


def test_footprint_and_selection_split_deep_tail():
    # distinct blocks * corpus block_size; repeated ids don't inflate footprint
    t_small = {"id": "a", "block_size": 64,
               "requests": [{"hash_ids": [1, 2, 3]}, {"hash_ids": [1, 2, 3, 4]}]}
    assert trace_footprint(t_small) == 4 * 64  # 4 distinct blocks
    t_big = {"id": "b", "block_size": 64, "requests": [{"hash_ids": list(range(1, 100))}]}
    kept, excluded = select_by_footprint([t_small, t_big], footprint_cap=64 * 10)
    assert [t["id"] for t in kept] == ["a"]
    assert [t["id"] for t in excluded] == ["b"]  # deep tail returned, not dropped silently
    # no cap -> everything kept, nothing excluded
    kept2, excluded2 = select_by_footprint([t_small, t_big], footprint_cap=None)
    assert len(kept2) == 2 and excluded2 == []


def test_build_workload_synthesizes_per_shard():
    wl = build_workload(_traces(6), n_instances=2, shared_prefix_blocks=2,
                        block_size=16, vocab_size=1000)
    assert set(wl) == {0, 1}
    # every session is a list of requests with synthesized token_ids
    for sessions in wl.values():
        for sess in sessions:
            assert sess and all("token_ids" in r for r in sess)
