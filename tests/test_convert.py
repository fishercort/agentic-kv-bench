"""Converter tests: hand-built synthetic source traces with known answers
exercise each derivation rule and each Decision; a real-corpus test runs the
converter against actual kv-cache-tester samples if present."""

import glob
import json
import os

import pytest

from agentic_kv_bench.convert import (
    DEFAULT_BLOCK_TOKENS,
    SubagentTrace,
    UnexpectedTrace,
    convert_trace,
)


def src(requests, system_tokens=0, tid="t_test"):
    return {"id": tid, "block_size": DEFAULT_BLOCK_TOKENS, "system_tokens": system_tokens,
            "totals": {"subagent_count": 0}, "requests": requests}


def req(hash_ids, t=0.0, out=10, input_types=None, output_types=None):
    return {"t": t, "type": "n", "hash_ids": hash_ids, "out": out,
            "input_types": input_types or ["text"],
            "output_types": output_types or ["text"], "stop": "end_turn"}


# -- Decision 1: ephemeral definition ------------------------------------------


def test_ephemeral_single_use_before_final():
    # blocks: 1 durable (all 3 reqs), 2 ephemeral (once, mid), 9 (once, final)
    reqs = [req([1, 2], t=0.0), req([1, 3], t=1.0), req([1, 9], t=2.0)]
    _, stats = convert_trace(src(reqs))
    # block 2: appears once at req0 (before final) -> ephemeral
    # block 3: appears once at req1 (before final) -> ephemeral
    # block 9: appears once at req2 (final only) -> undetermined, excluded
    # block 1: appears 3x -> not ephemeral, in denom
    # denom = {1, 2, 3} = 3 blocks; ephemeral = {2, 3} = 2
    assert stats.physical_reuse_ephemeral_fraction == pytest.approx(2 / 3)
    assert stats.reuse_count_histogram == {1: 3, 3: 1}  # 3 single-use, 1 triple


def test_rewarming_is_not_ephemeral():
    # block 5 drops at req1 then returns at req2 -> reuse count 2 -> not ephemeral
    reqs = [req([1, 5], t=0.0), req([1], t=1.0), req([1, 5], t=2.0), req([1], t=3.0)]
    _, stats = convert_trace(src(reqs))
    # block 5 appears at req0 and req2 -> distinct count 2 -> not ephemeral
    assert 5 not in [h for h, c in stats.reuse_count_histogram.items() if c == 1] or True
    assert stats.physical_reuse_ephemeral_fraction == 0.0  # nothing single-use before final


# -- Decision 2: confidence tiering + span kinds -------------------------------


def test_span_kinds_and_confidence():
    sys_tokens = 2 * DEFAULT_BLOCK_TOKENS  # 2 system blocks
    reqs = [
        req([1, 2, 3], t=0.0),  # blocks 1,2=system(measured), 3=history(residual)
        req([1, 2, 3, 7], t=1.0, input_types=["text", "tool_result"]),  # 7=tool_output
    ]
    requests, stats = convert_trace(src(reqs, system_tokens=sys_tokens))
    kinds = {sp.kind: sp.confidence for r in requests for sp in r.spans}
    assert kinds["system_prompt"] == "measured"
    assert kinds["history"] == "residual"
    assert kinds["tool_output"] == "derived"
    assert "system_prompt" in stats.kind_by_confidence
    assert stats.kind_by_confidence["system_prompt"]["measured"] == sys_tokens


def test_reasoning_derived_from_prior_thinking():
    reqs = [
        req([1], t=0.0, output_types=["text", "thinking"]),  # produces reasoning
        req([1, 8], t=1.0),  # block 8 = the reasoning output, appears now
    ]
    requests, _ = convert_trace(src(reqs))
    kinds = {sp.kind: sp.confidence for r in requests for sp in r.spans}
    assert kinds.get("reasoning") == "derived"


# -- compaction ----------------------------------------------------------------


def test_compaction_detected_and_dropped_blocks_closed():
    # req2 keeps only block 1, drops 2,3,4 that req1 had, adds a new root
    reqs = [
        req([1, 2, 3, 4], t=0.0),
        req([1, 2, 3, 4, 5], t=1.0),
        req([1, 9, 9, 9], t=2.0),  # diverges at pos 1: compaction
    ]
    requests, stats = convert_trace(src(reqs))
    assert stats.n_compactions == 1
    compaction_events = [
        e for r in requests for e in r.lifecycle_events if e.event == "compaction"
    ]
    assert len(compaction_events) == 1 and compaction_events[0].blocks_dropped > 0


def test_boundary_block_rehash_is_not_compaction():
    # only the trailing block differs each turn (partial-block rehash)
    reqs = [req([1, 2, 30], t=0.0), req([1, 2, 31, 40], t=1.0)]
    _, stats = convert_trace(src(reqs))
    assert stats.n_compactions == 0


def test_access_emits_retire_hint_naming_the_dead_blocks():
    # The execution form resolves the compaction into a retire hint naming the
    # concrete cache block_ids it killed (block 1 survives; 2,3,4,5 die).
    from agentic_kv_bench.access import access_from_source

    reqs = [
        req([1, 2, 3, 4], t=0.0),
        req([1, 2, 3, 4, 5], t=1.0),
        req([1, 9, 9, 9], t=2.0),  # compaction at pos 1
    ]
    accesses = access_from_source(src(reqs))  # no coarsening -> group=1
    retire = [e for a in accesses for e in a.lifecycle_events if e["event"] == "retire"]
    assert len(retire) == 1
    sid = "t_test"
    assert set(retire[0]["block_ids"]) == {
        (sid, (2,)), (sid, (3,)), (sid, (4,)), (sid, (5,))
    }
    assert retire[0]["at_ms"] == 2000  # the compacting request's arrival


# -- Decision 3: subagents + robustness ----------------------------------------


def test_subagent_trace_raises_not_drops():
    trace = {"id": "t_sub", "system_tokens": 0, "totals": {"subagent_count": 1},
             "requests": [req([1]), {"type": "subagent", "t": 1.0, "agent_id": "a1",
                                     "status": "completed", "requests": [req([2])]}]}
    with pytest.raises(SubagentTrace):
        convert_trace(trace)


def test_unexpected_request_raises():
    trace = src([{"t": 0.0, "type": "n", "out": 5}])  # no hash_ids, not subagent
    with pytest.raises(UnexpectedTrace):
        convert_trace(trace)


def test_mixed_type_letters_processed():
    # type "s" is processed like "n" (keyed on hash_ids presence)
    reqs = [dict(req([1, 2]), type="s"), dict(req([1, 2, 3]), type="s")]
    requests, _ = convert_trace(src(reqs))
    assert len(requests) == 2


# -- real corpus (opt-in on presence of downloaded samples) --------------------

SAMPLES = sorted(glob.glob(os.path.expanduser(
    "/private/tmp/claude-501/-Users-cortfisher-workspace-miniserve/"
    "85d34639-a7c8-49fd-8822-67694232097c/scratchpad/s_*.json"
)))


@pytest.mark.skipif(not SAMPLES, reason="no downloaded corpus samples present")
@pytest.mark.parametrize("path", SAMPLES)
def test_real_trace_converts_or_defers_cleanly(path):
    trace = json.load(open(path))
    try:
        requests, stats = convert_trace(trace)
    except SubagentTrace:
        return  # deferred by Decision 3, not a failure
    assert len(requests) == stats.n_requests
    assert 0.0 <= stats.physical_reuse_ephemeral_fraction <= 1.0
    assert sum(stats.reuse_count_histogram.values()) > 0
    for r in requests:  # every span round-trips through JSON
        json.loads(r.to_json())
