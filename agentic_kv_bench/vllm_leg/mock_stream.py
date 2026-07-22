"""Version-pinned mock of the vLLM KV-event stream (docs/vllm-leg-design.md §6).

The agent is coded and unit-tested against THIS recording, not a live GPU. Events are
in the exact `array_like` + tag-first wire form vLLM's msgpack encoder produces (see
kv_events.decode_batch), so the only thing separating this from the box is the ZMQ
transport. The on-box minute-one check decodes a live batch, takes `wire_fingerprint`,
and diffs it against GOLDEN_FINGERPRINT: schema drift is a diff, not a mystery.

Provenance is pinned to the same vLLM ref as kv_events. Block hashes here are small ints
for readability; on the box they are 32-byte sha256 (ExternalBlockHash = Union[bytes,
int]) and the agent treats them opaquely, so the shape is identical. The one thing this
mock CANNOT capture is the NONE_HASH/PYTHONHASHSEED cross-instance seed agreement (§6) —
that is a live-only sanity check, called out in the runbook.
"""

from .kv_events import EVENT_SCHEMA_VERSION, VLLM_SOURCE_REF, decode_batch

BLOCK_SIZE = 4

# Two instances (each its own ZMQ endpoint). Instance 1's request shares a 2-block
# prefix (hashes 101, 102) with instance 0's already-resident request, so a
# cross-instance-aware cache would have avoided 2 blocks of recompute on instance 1.
# Hash 104 is unique to instance 1. This is the known-answer scenario for the agent test:
# residual = 2 blocks * BLOCK_SIZE = 8 tokens.
#
# Wire form per event (tag first, then fields in declared order):
#   BlockStored:     ["BlockStored", block_hashes, parent, token_ids, block_size, lora, medium]
#   BlockRemoved:    ["BlockRemoved", block_hashes, medium]
#   AllBlocksCleared:["AllBlocksCleared", medium]
# Batch: [ts, [event, ...], data_parallel_rank]

_INSTANCE_0 = [
    [1.0, [
        ["BlockStored", [101, 102, 103], None, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
         BLOCK_SIZE, None, "GPU"],
    ], 0],
]

_INSTANCE_1 = [
    [2.0, [
        # shares prefix 101,102 with instance 0 (resident since t=1.0); 104 is new.
        ["BlockStored", [101, 102, 104], None, [0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23],
         BLOCK_SIZE, None, "GPU"],
    ], 0],
    [3.0, [
        ["BlockRemoved", [104], "GPU"],       # exercise remove decode path
    ], 0],
    [4.0, [
        ["AllBlocksCleared", "GPU"],          # clear decode path, medium present
    ], 0],
    [5.0, [
        ["AllBlocksCleared"],                 # exercise omit_defaults trailing-None drop
    ], 0],
]

GOLDEN = {
    "vllm_source_ref": VLLM_SOURCE_REF,
    "event_schema_version": EVENT_SCHEMA_VERSION,
    "block_size": BLOCK_SIZE,
    "streams": {0: _INSTANCE_0, 1: _INSTANCE_1},
    # per-request local APC hits (from Prometheus num_local_cached_tokens), keyed by the
    # BlockStored we correlate to. Instance 1 caught 0 locally (cold), so all shared
    # blocks are residual. See agent.py for the correlation.
    "num_local_cached_tokens": {0: {1.0: 0}, 1: {2.0: 0}},
    "expected_residual_tokens": 8,  # 2 shared blocks * block_size
}


def golden_batches(instance_id):
    """Decoded KVEventBatch list for one instance (proves decode_batch round-trips the
    recorded wire form)."""
    return [decode_batch(b) for b in GOLDEN["streams"][instance_id]]


def wire_fingerprint(batch_arr):
    """Structural signature of a raw wire batch: (batch_arity, [(tag, event_arity), ...]).
    Value-independent — this is what the on-box live stream is diffed against so a field
    added/removed/reordered by a vLLM bump shows up immediately."""
    events = batch_arr[1]
    return (len(batch_arr), tuple((e[0], len(e)) for e in events))


GOLDEN_FINGERPRINT = {
    inst: [wire_fingerprint(b) for b in batches]
    for inst, batches in GOLDEN["streams"].items()
}
