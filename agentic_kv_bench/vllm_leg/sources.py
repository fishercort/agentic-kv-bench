"""Source adapters: map an engine's native cache telemetry to the normalized event model
(event_model). Compute consumes only normalized events, so a new engine is a new Source with
zero downstream change — the multi-engine story as an interface, not a slide.

vLLM is the reference adapter. Content is already dropped at the vLLM decode boundary
(kv_events sets token_ids=None); normalization carries no content field at all, so the
metadata-only property is structural here too."""

from collections.abc import Iterator
from typing import Protocol

from .event_model import ClearedAll, RemovedBlock, StoredBlock


class Source(Protocol):
    """Yields (source_id, normalized_event). source_id identifies the engine instance for
    cross-instance analysis. Implementations wrap a transport (ZMQ, file, ...) + an adapter."""

    def events(self) -> Iterator[tuple]:
        ...


def normalize_vllm_batch(source_id, batch):
    """vLLM KVEventBatch -> normalized events. A vLLM BlockStored carries a LIST of block
    hashes; it fans out to one StoredBlock per hash. token_ids are already None (dropped at
    decode) and have no representation in the normalized model."""
    for ev in batch.events:
        tag = type(ev).__name__
        if tag == "BlockStored":
            for h in ev.block_hashes:
                yield source_id, StoredBlock(h, ev.parent_block_hash, ev.block_size,
                                             ev.medium, batch.ts)
        elif tag == "BlockRemoved":
            for h in ev.block_hashes:
                yield source_id, RemovedBlock(h, ev.medium, batch.ts)
        elif tag == "AllBlocksCleared":
            yield source_id, ClearedAll(ev.medium, batch.ts)
