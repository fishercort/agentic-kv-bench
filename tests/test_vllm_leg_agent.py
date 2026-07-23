"""Telemetry-agent acceptance tests, run with zero GPU time against the version-pinned
mock."""

import pytest

from agentic_kv_bench.vllm_leg.agent import (
    TelemetryAgent,
    assert_metadata_only,
    salted_id,
)
from agentic_kv_bench.vllm_leg.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    SchemaDriftError,
    decode_batch,
    decode_event,
)
from agentic_kv_bench.vllm_leg.mock_stream import (
    GOLDEN,
    GOLDEN_FINGERPRINT,
    golden_batches,
    wire_fingerprint,
)

PROV = {
    "gpu": "L40S", "driver": "550.x", "vllm_version": "0.11.0",
    "event_schema_version": GOLDEN["event_schema_version"],
    "dtype": "bf16", "block_size": GOLDEN["block_size"],
}


# --- schema decode / drift ----------------------------------------------------
def test_decode_roundtrips_golden_wire_form():
    b = golden_batches(0)[0]
    assert isinstance(b.events[0], BlockStored)
    assert b.events[0].block_hashes == [101, 102, 103]
    assert b.events[0].block_size == 4
    assert b.events[0].medium == "GPU"


def test_omit_defaults_trailing_medium_is_tolerated():
    # ["AllBlocksCleared"] with medium dropped by omit_defaults must decode with None.
    ev = decode_event(["AllBlocksCleared"])
    assert isinstance(ev, AllBlocksCleared) and ev.medium is None
    ev2 = decode_event(["BlockRemoved", [104]])
    assert isinstance(ev2, BlockRemoved) and ev2.medium is None


def test_decode_drops_token_content_on_ingest():
    # vLLM's BlockStored carries token_ids (prompt tokens); the agent must NEVER retain them.
    # Dropped at the decode boundary -> the metadata-only guarantee (--show-payload verifies).
    ev = decode_event(["BlockStored", [1, 2], None, [10, 11, 12, 13, 14, 15, 16, 17],
                       4, None, "GPU"])
    assert isinstance(ev, BlockStored)
    assert ev.block_hashes == [1, 2] and ev.block_size == 4 and ev.medium == "GPU"  # metadata kept
    assert ev.token_ids is None  # token content dropped, never held


def test_unknown_tag_is_schema_drift():
    with pytest.raises(SchemaDriftError):
        decode_event(["BlockRelocated", [1], "GPU"])


def test_wrong_arity_is_schema_drift():
    # a BlockStored missing a required field (would silently mis-map without the guard)
    with pytest.raises(SchemaDriftError):
        decode_event(["BlockStored", [1], None])


def test_wire_fingerprint_matches_golden_and_catches_drift():
    for inst in GOLDEN["streams"]:
        live = [wire_fingerprint(b) for b in GOLDEN["streams"][inst]]
        assert live == GOLDEN_FINGERPRINT[inst]
    # a field appended by a hypothetical vLLM bump changes the fingerprint -> visible diff
    bumped = ["BlockStored", [1], None, [0, 1, 2, 3], 4, None, "GPU", "new_field"]
    assert wire_fingerprint([1.0, [bumped], 0]) != GOLDEN_FINGERPRINT[0][0]


# --- residual dedup -----------------------------------------------------------
def test_residual_matches_known_answer():
    agent = TelemetryAgent(PROV, salt="deploy-salt", block_size=GOLDEN["block_size"])
    streams = {inst: golden_batches(inst) for inst in GOLDEN["streams"]}
    total = agent.run(streams)
    assert total == GOLDEN["expected_residual_tokens"]  # 2 shared blocks * 4 = 8


def test_residual_is_zero_without_cross_instance_sharing():
    # instance 1's blocks are disjoint -> no residual, even though both are busy.
    a = decode_batch([1.0, [["BlockStored", [1, 2], None, [0, 1, 2, 3, 4, 5, 6, 7],
                             4, None, "GPU"]], 0])
    b = decode_batch([2.0, [["BlockStored", [8, 9], None, [0, 1, 2, 3, 4, 5, 6, 7],
                             4, None, "GPU"]], 0])
    agent = TelemetryAgent(PROV, salt="s", block_size=4)
    assert agent.run({0: [a], 1: [b]}) == 0


def test_eviction_before_reuse_is_not_residual():
    # instance 0 stores then clears block 5 BEFORE instance 1 stores it -> not resident
    # elsewhere at store time -> not residual. Guards the as-of-time semantics.
    s0 = decode_batch([1.0, [["BlockStored", [5], None, [0, 1, 2, 3], 4, None, "GPU"]], 0])
    c0 = decode_batch([2.0, [["AllBlocksCleared", "GPU"]], 0])
    s1 = decode_batch([3.0, [["BlockStored", [5], None, [0, 1, 2, 3], 4, None, "GPU"]], 0])
    agent = TelemetryAgent(PROV, salt="s", block_size=4)
    assert agent.run({0: [s0, c0], 1: [s1]}) == 0


# --- day-one schema: salting, metadata-only, lifecycle ------------------------
def test_salting_preserves_cross_instance_overlap_but_hides_raw():
    # same raw hash -> same salted id (else overlap detection breaks after salting);
    # different domain or salt -> different id (domain isolation).
    assert salted_id(101, "d", "salt") == salted_id(101, "d", "salt")
    assert salted_id(101, "d", "salt") != salted_id(102, "d", "salt")
    assert salted_id(101, "d1", "salt") != salted_id(101, "d2", "salt")
    assert salted_id(101, "d", "salt1") != salted_id(101, "d", "salt2")
    assert salted_id(101, "d", "salt") != "101"  # raw id never appears


def test_records_are_metadata_only():
    agent = TelemetryAgent(PROV, salt="s", block_size=GOLDEN["block_size"])
    agent.run({inst: golden_batches(inst) for inst in GOLDEN["streams"]})
    assert agent.records
    for r in agent.records:
        assert_metadata_only(r)  # raises if any record leaks content
        assert r["event_schema_version"] == GOLDEN["event_schema_version"]


def test_metadata_only_guard_rejects_token_content():
    with pytest.raises(ValueError):
        assert_metadata_only({"kind": "x", "token_ids": [1, 2, 3, 4]})
    with pytest.raises(ValueError):
        assert_metadata_only({"kind": "x", "sneaky": [10, 11, 12, 13, 14]})


def test_lifecycle_records_emitted_for_remove_and_clear():
    agent = TelemetryAgent(PROV, salt="s", block_size=GOLDEN["block_size"])
    agent.run({inst: golden_batches(inst) for inst in GOLDEN["streams"]})
    reasons = {r["reason"] for r in agent.records if r["kind"] == "lifecycle"}
    assert "evict" in reasons and "clear" in reasons


# --- eviction waste (number two) ----------------------------------------------
def test_eviction_recompute_counts_restore_after_evict():
    # store block 7, evict it, store it again on the SAME instance -> one eviction recompute
    s1 = decode_batch([1.0, [["BlockStored", [7], None, [0, 1, 2, 3], 4, None, "GPU"]], 0])
    rm = decode_batch([2.0, [["BlockRemoved", [7], "GPU"]], 0])
    s2 = decode_batch([3.0, [["BlockStored", [7], None, [0, 1, 2, 3], 4, None, "GPU"]], 0])
    agent = TelemetryAgent(PROV, salt="s", block_size=4)
    for b in (s1, rm, s2):
        agent.ingest_batch(0, b)
    assert agent.eviction_recompute_tokens == 4  # one block re-stored after eviction
    kinds = {r["kind"] for r in agent.records}
    assert "eviction_recompute" in kinds


def test_store_without_prior_evict_is_not_eviction_recompute():
    # a first-time store (never evicted) is not waste
    s1 = decode_batch([1.0, [["BlockStored", [7, 8], None, [0, 1, 2, 3, 4, 5, 6, 7],
                              4, None, "GPU"]], 0])
    agent = TelemetryAgent(PROV, salt="s", block_size=4)
    agent.ingest_batch(0, s1)
    assert agent.eviction_recompute_tokens == 0


def test_record_cap_bounds_ram_but_counts_stay_exact():
    # many residual blocks; cap the in-memory sample at 1 but counts must be exact.
    a = decode_batch([1.0, [["BlockStored", [1, 2, 3], None, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                             4, None, "GPU"]], 0])
    b = decode_batch([2.0, [["BlockStored", [1, 2, 3], None, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                             4, None, "GPU"]], 0])
    agent = TelemetryAgent(PROV, salt="s", block_size=4, record_cap=1)
    agent.ingest_batch(0, a)
    agent.ingest_batch(1, b)  # 3 cross-instance residual blocks
    assert len(agent.records) == 1                       # RAM bounded
    assert agent.record_counts.get("residual_block") == 3  # count exact
    assert agent.residual_tokens == 12                   # 3 blocks * 4, exact


def test_catalog_stats_report_footprint_drivers():
    agent = TelemetryAgent(PROV, salt="s", block_size=4)
    agent.run({inst: golden_batches(inst) for inst in GOLDEN["streams"]})
    st = agent.catalog_stats()
    assert st["catalog_peak_entries"] >= st["tracked_current"]
    assert "events_ingested" in st and st["events_ingested"] > 0
    assert set(st["record_counts"]) <= {"residual_block", "eviction_recompute", "lifecycle"}


def test_eviction_recompute_is_within_instance_only():
    # block resident on instance 0, evicted there; instance 1 storing it is RESIDUAL
    # (cross-instance), not eviction recompute on instance 1 (never evicted there).
    a = decode_batch([1.0, [["BlockStored", [5], None, [0, 1, 2, 3], 4, None, "GPU"]], 0])
    b = decode_batch([2.0, [["BlockStored", [5], None, [0, 1, 2, 3], 4, None, "GPU"]], 0])
    agent = TelemetryAgent(PROV, salt="s", block_size=4)
    agent.ingest_batch(0, a)
    agent.ingest_batch(1, b)
    assert agent.eviction_recompute_tokens == 0   # neither instance re-stored post-evict
    assert agent.residual_tokens == 4             # instance 1's store is cross-instance
