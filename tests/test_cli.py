"""Adapter + CLI tests: the adapter reconstructs correct block-level access,
the CLI convert/run/oracle verbs work end to end, and a policy loads by import
path (the adoption surface, exercised through the actual command)."""

import glob
import json

import pytest

from agentic_kv_bench.access import access_from_source
from agentic_kv_bench.cli import load_policy, main
from agentic_kv_bench.convert import SubagentTrace


def src(requests, system_tokens=0, tid="t"):
    return {"id": tid, "block_size": 64, "system_tokens": system_tokens,
            "totals": {"subagent_count": 0}, "requests": requests}


def req(hash_ids, t=0.0, out=5):
    return {"t": t, "type": "n", "hash_ids": hash_ids, "out": out,
            "input_types": ["text"], "output_types": ["text"], "stop": "end_turn"}


def test_adapter_reconstructs_block_access():
    trace = src([req([1, 2]), req([1, 2, 3])])
    accesses = access_from_source(trace)
    assert [len(a.blocks) for a in accesses] == [2, 3]
    assert accesses[1].blocks[2].block_id == "t#3"  # namespaced by session id
    assert all(b.size_tokens == 64 for a in accesses for b in a.blocks)
    # block kinds match the converter's derivation (shared walk_source)
    assert accesses[0].blocks[0].kind == "history"


def test_adapter_defers_subagents():
    trace = {"id": "s", "system_tokens": 0, "totals": {"subagent_count": 1},
             "requests": [req([1]), {"type": "subagent", "t": 1.0, "requests": [req([2])]}]}
    with pytest.raises(SubagentTrace):
        access_from_source(trace)


def test_load_policy_resolves_and_validates():
    cls = load_policy("agentic_kv_bench.baselines:LRU")
    from agentic_kv_bench.baselines import LRU
    assert cls is LRU
    with pytest.raises(SystemExit):
        load_policy("no_colon")
    with pytest.raises(SystemExit):
        load_policy("agentic_kv_bench.baselines:NotAPolicy")


def test_cli_convert_and_run_end_to_end(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Two sessions. Each fits alone in 2 blocks, but interleaved they compete
    # for a 2-block cache -> real cross-session eviction and nonzero cost.
    (corpus / "a.json").write_text(json.dumps(
        src([req([1, 2]), req([1, 2])], tid="a")))
    (corpus / "b.json").write_text(json.dumps(
        src([req([1, 2]), req([1, 2])], tid="b")))

    out = tmp_path / "traces.jsonl"
    main(["convert", str(corpus), "-o", str(out)])
    assert out.exists() and sum(1 for _ in out.open()) == 4  # 2 sessions x 2 reqs

    main(["run", str(corpus), "--policy", "agentic_kv_bench.baselines:LRU",
          "--capacity-tokens", "128", "--session-gap-ms", "0"])  # 2 blocks, overlapping
    report = capsys.readouterr().out
    assert "PERCENT OF ORACLE" in report
    assert "2 sessions interleaved" in report
    assert "policy:   agentic_kv_bench.baselines:LRU" in report


def test_cli_oracle_verb(tmp_path, capsys):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.json").write_text(json.dumps(src([req([1, 2]), req([1, 2])], tid="a")))
    (corpus / "b.json").write_text(json.dumps(src([req([3, 4]), req([3, 4])], tid="b")))
    main(["oracle", str(corpus), "--capacity-tokens", "128", "--session-gap-ms", "0"])
    assert "oracle scored cost" in capsys.readouterr().out


# -- real corpus end-to-end (opt-in on downloaded samples) ---------------------

SAMPLES = sorted(glob.glob(
    "/private/tmp/claude-501/-Users-cortfisher-workspace-miniserve/"
    "85d34639-a7c8-49fd-8822-67694232097c/scratchpad/s_*.json"
))


@pytest.mark.skipif(not SAMPLES, reason="no downloaded corpus samples")
@pytest.mark.parametrize("path", SAMPLES)
def test_real_trace_single_session_feasible(path):
    """One real session at capacity >= its largest prefix replays feasibly and
    the plumbing (adapter -> harness -> oracle) works on real data. Note it may
    still evict: compaction leaves dropped blocks resident until they age out,
    so total resident can exceed any single prefix. The robust invariants are
    feasibility, the oracle lower bound, and balanced accounting."""
    trace = json.load(open(path))
    try:
        accesses = access_from_source(trace)
    except SubagentTrace:
        return
    from agentic_kv_bench.baselines import LRU
    from agentic_kv_bench.harness import CostParams, replay
    from agentic_kv_bench.oracle import oracle_run, percent_of_oracle

    max_prefix = max(sum(b.size_tokens for b in a.blocks) for a in accesses)
    cost = CostParams()
    res = replay(accesses, LRU(), cost, max_prefix)
    ora = oracle_run(accesses, cost, max_prefix)
    assert percent_of_oracle(res, ora) >= 100.0 - 1e-6  # oracle is the lower bound
    assert res.hits + res.compulsory_misses + res.capacity_misses == res.total_accesses


@pytest.mark.skipif(len(SAMPLES) < 2, reason="need >= 2 corpus samples")
def test_real_multisession_pressure_and_lower_bound():
    """Interleave real sessions in a cache smaller than their combined working
    sets: real eviction, and the oracle remains the lower bound on real data."""
    from agentic_kv_bench.baselines import LRU
    from agentic_kv_bench.harness import CostParams, interleave, replay
    from agentic_kv_bench.oracle import oracle_run, percent_of_oracle

    sessions = []
    for p in SAMPLES:
        try:
            sessions.append(access_from_source(json.load(open(p))))
        except SubagentTrace:
            continue
    sessions = sessions[:3]
    merged = interleave(sessions, gap_ms=0)  # fully overlapping = max pressure
    largest = max(sum(b.size_tokens for b in a.blocks) for s in sessions for a in s)
    cap = largest + 64  # fits any single request, not all sessions at once
    cost = CostParams()
    res = replay(merged, LRU(), cost, cap)
    ora = oracle_run(merged, cost, cap)
    assert res.n_evictions > 0  # genuine cross-session pressure
    assert percent_of_oracle(res, ora) >= 100.0 - 1e-6
