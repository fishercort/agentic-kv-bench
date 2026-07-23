"""Normalized KV telemetry event model — the Python reference of the v0 spec
(kv-policy-core/docs/telemetry-event-model.md). It is the contract between a Source (engine
adapter) and Compute (analysis): a Source maps an engine's native cache events to these,
Compute consumes only these, so Compute is engine-independent.

Metadata-only by design: no event has a field that could carry token content. One block per
event (a vLLM BlockStored with a list of hashes fans out to N StoredBlock at the Source)."""

from dataclasses import dataclass

SCHEMA_VERSION = "v0"


@dataclass(frozen=True)
class StoredBlock:
    """A full block was (re)computed and inserted into the cache."""
    block_hash: object          # prefix-chained; salted before egress
    parent_hash: object | None  # chained parent (None = prefix root)
    block_size: int             # tokens per block (cache granularity)
    medium: str | None          # GPU | CPU | FS | OBJ
    ts: float


@dataclass(frozen=True)
class RemovedBlock:
    """A block was evicted/dropped from a tier."""
    block_hash: object
    medium: str | None
    ts: float


@dataclass(frozen=True)
class ClearedAll:
    """A tier was fully cleared (flush, restart, session close)."""
    medium: str | None
    ts: float
