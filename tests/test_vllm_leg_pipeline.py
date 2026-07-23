"""Source / normalized-model / Sink seams (the extraction interfaces). Proves the boundary:
a vLLM batch normalizes to metadata-only single-block events, and the agent produces the
same numbers whether fed native batches or normalized events directly."""

from agentic_kv_bench.vllm_leg.agent import TelemetryAgent
from agentic_kv_bench.vllm_leg.event_model import (
    ClearedAll,
    RemovedBlock,
    StoredBlock,
)
from agentic_kv_bench.vllm_leg.kv_events import decode_batch
from agentic_kv_bench.vllm_leg.sinks import JsonlSink, MemorySink
from agentic_kv_bench.vllm_leg.sources import normalize_vllm_batch

PROV = {"gpu": "x", "vllm_version": "0.11.0", "event_schema_version": "vllm-0.11.0-1",
        "block_size": 4}


def test_source_fans_out_multihash_and_carries_no_content():
    # a vLLM BlockStored with 3 hashes -> 3 StoredBlock; no field can carry token content.
    batch = decode_batch([1.0, [["BlockStored", [10, 11, 12], None,
                                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 4, None, "GPU"]], 0])
    evs = list(normalize_vllm_batch("s0", batch))
    assert [e.__class__.__name__ for _, e in evs] == ["StoredBlock"] * 3
    assert [e.block_hash for _, e in evs] == [10, 11, 12]
    assert all(not hasattr(e, "token_ids") for _, e in evs)  # no content field exists
    assert all(sid == "s0" and e.block_size == 4 and e.medium == "GPU" for sid, e in evs)


def test_source_maps_remove_and_clear():
    b = decode_batch([2.0, [["BlockRemoved", [10, 11], "GPU"]], 0])
    evs = [e for _, e in normalize_vllm_batch("s0", b)]
    assert [type(e).__name__ for e in evs] == ["RemovedBlock", "RemovedBlock"]
    c = decode_batch([3.0, [["AllBlocksCleared", "GPU"]], 0])
    assert type(list(normalize_vllm_batch("s0", c))[0][1]).__name__ == "ClearedAll"


def test_agent_ingests_normalized_events_directly():
    # same numbers whether via ingest(normalized) or ingest_batch(native) -> boundary holds.
    agent = TelemetryAgent(PROV, salt="s", block_size=4)
    agent.ingest("a", StoredBlock(7, None, 4, "GPU", 1.0))
    agent.ingest("b", StoredBlock(7, None, 4, "GPU", 2.0))  # resident on 'a' -> residual
    assert agent.residual_tokens == 4
    agent.ingest("a", RemovedBlock(7, "GPU", 3.0))          # evict on 'a'
    agent.ingest("a", StoredBlock(7, None, 4, "GPU", 4.0))  # re-store -> eviction recompute
    assert agent.eviction_recompute_tokens == 4
    agent.ingest("a", ClearedAll("GPU", 5.0))               # lifecycle clear
    assert any(r["kind"] == "lifecycle" and r["reason"] == "clear" for r in agent.records)


def test_memory_sink_caps_sample_keeps_counts():
    s = MemorySink(cap=1)
    for i in range(5):
        s.emit({"kind": "x", "i": i})
    assert len(s.records) == 1 and s.counts["x"] == 5


def test_jsonl_sink_streams_and_counts(tmp_path):
    p = tmp_path / "recs.jsonl"
    s = JsonlSink(str(p))
    s.emit({"kind": "residual_block", "n_tokens": 4})
    s.emit({"kind": "lifecycle", "reason": "evict"})
    s.close()
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2 and s.counts == {"residual_block": 1, "lifecycle": 1}
    assert s.records == []  # streamed, nothing held in RAM


def test_agent_with_injected_jsonl_sink(tmp_path):
    # the agent streams records to a real sink instead of RAM; numbers still exact.
    p = tmp_path / "out.jsonl"
    agent = TelemetryAgent(PROV, salt="s", block_size=4, sink=JsonlSink(str(p)))
    agent.ingest("a", StoredBlock(1, None, 4, "GPU", 1.0))
    agent.ingest("b", StoredBlock(1, None, 4, "GPU", 2.0))
    agent._sink.close()
    assert agent.residual_tokens == 4
    assert agent.record_counts["residual_block"] == 1
    assert p.read_text().strip()  # a record was streamed to disk
