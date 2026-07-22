"""vLLM KV-cache event schema, version-pinned.

Transcribed and VERIFIED against vLLM source (see VLLM_SOURCE_REF) so the telemetry
agent can be coded and unit-tested with zero GPU time. The on-box minute-one check
diffs a live decoded batch against SCHEMA_FINGERPRINT; drift surfaces as a diff, not a
mystery (docs/vllm-leg-design.md §6).

Wire format (ZmqEventPublisher, v0.11.0): PUB socket, 3-part multipart message
    (topic_bytes, seq.to_bytes(8, "big"), msgpack_payload)
The payload is a msgspec msgpack encoding of a KVEventBatch. The structs are
`array_like=True` (each struct encodes as an ARRAY, not a map) and the event union is
`tag=True` (each event array's first element is its tag = the class name). So on the
wire:
    batch  = [ts, [event, ...], data_parallel_rank]
    event  = ["BlockStored", block_hashes, parent_block_hash, token_ids,
              block_size, lora_id, medium]   # tag-first, field order below
`decode_batch` reconstructs these dataclasses from that array form; it is the only
layer that touches the wire encoding, so it is the only thing the on-box schema diff
has to validate.

Operational trap (verified in kv_cache_utils.py, load-bearing for residual dedup):
`ExternalBlockHash = Union[bytes, int]`, chained by `hash_fn((parent, tokens, extra))`,
and `NONE_HASH` (the seed of every prefix's first block) is `os.urandom(32)` UNLESS
`PYTHONHASHSEED` is set. Two instances with different NONE_HASH hash the SAME prefix
differently, so cross-instance overlap reads as zero. The runbook pins the same
PYTHONHASHSEED + sha256 on both instances; the agent assumes that pin holds and asserts
it in the minute-one check (see vllm-leg-design.md §6).
"""

from dataclasses import dataclass, field

# --- provenance / version pin -------------------------------------------------
VLLM_SOURCE_REF = "vllm-project/vllm@v0.11.0:vllm/distributed/kv_events.py"
VLLM_HASH_UTILS_REF = "vllm-project/vllm@v0.11.0:vllm/v1/core/kv_cache_utils.py"
VERIFIED_ON = "2026-07-22"  # date the schema above was read from source
# Bump this when the transcribed schema below changes. Stamped into every emitted
# record as `event_schema_version` (day-one stack provenance).
EVENT_SCHEMA_VERSION = "vllm-0.11.0-1"

# ExternalBlockHash = Union[bytes, int] in vLLM (sha256 bytes by default; int for
# backward compat). The agent treats hashes opaquely, so either flows unchanged.
ExternalBlockHash = bytes | int


# --- event structs (field ORDER matters: it is the array_like wire order) ------
@dataclass(frozen=True)
class BlockStored:
    block_hashes: list  # list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list  # list[int]
    block_size: int
    lora_id: int | None
    medium: str | None = None
    TAG = "BlockStored"


@dataclass(frozen=True)
class BlockRemoved:
    block_hashes: list  # list[ExternalBlockHash]
    medium: str | None = None
    TAG = "BlockRemoved"


@dataclass(frozen=True)
class AllBlocksCleared:
    medium: str | None = None
    TAG = "AllBlocksCleared"


@dataclass(frozen=True)
class KVEventBatch:
    ts: float
    events: list = field(default_factory=list)
    data_parallel_rank: int | None = None


# --- the fingerprint the on-box diff compares against -------------------------
# (tag -> ordered field names). If the live wire arrays don't match these arities
# and tags, the schema drifted and the run aborts rather than mis-decoding.
SCHEMA_FINGERPRINT = {
    "batch_fields": ["ts", "events", "data_parallel_rank"],
    "events": {
        "BlockStored": [
            "block_hashes", "parent_block_hash", "token_ids",
            "block_size", "lora_id", "medium",
        ],
        "BlockRemoved": ["block_hashes", "medium"],
        "AllBlocksCleared": ["medium"],
    },
}

_EVENT_TYPES = {
    "BlockStored": BlockStored,
    "BlockRemoved": BlockRemoved,
    "AllBlocksCleared": AllBlocksCleared,
}


class SchemaDriftError(ValueError):
    """Raised when a wire array does not match SCHEMA_FINGERPRINT. On the box this is
    the minute-one abort, not an hour-three silent mis-decode."""


def decode_event(arr):
    """Reconstruct one event from its tag-first array form (msgspec array_like+tag).
    ``arr[0]`` is the tag; the rest are fields in declared order. medium is optional
    (omit_defaults may drop trailing Nones), so short-by-one is tolerated and filled."""
    if not isinstance(arr, (list, tuple)) or not arr:
        raise SchemaDriftError(f"event is not a non-empty array: {arr!r}")
    tag = arr[0]
    cls = _EVENT_TYPES.get(tag)
    if cls is None:
        raise SchemaDriftError(f"unknown event tag {tag!r}; schema drift vs {VLLM_SOURCE_REF}")
    fields = SCHEMA_FINGERPRINT["events"][tag]
    payload = list(arr[1:])
    # omit_defaults can drop the trailing optional `medium`; pad it back to None.
    if len(payload) == len(fields) - 1 and fields[-1] == "medium":
        payload.append(None)
    if len(payload) != len(fields):
        raise SchemaDriftError(
            f"{tag} arity {len(payload)} != expected {len(fields)} "
            f"({fields}); schema drift vs {VLLM_SOURCE_REF}"
        )
    return cls(*payload)


def decode_batch(arr):
    """Reconstruct a KVEventBatch from its array form [ts, [event...], dp_rank?]."""
    if not isinstance(arr, (list, tuple)) or len(arr) < 2:
        raise SchemaDriftError(f"batch is not a [ts, events, ...] array: {arr!r}")
    ts = arr[0]
    events = [decode_event(e) for e in arr[1]]
    dp_rank = arr[2] if len(arr) > 2 else None
    return KVEventBatch(ts=ts, events=events, data_parallel_rank=dp_rank)
