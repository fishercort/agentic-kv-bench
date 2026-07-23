"""The telemetry agent's Compute module: consumes the normalized event model (event_model)
from any Source and produces two numbers + metadata-only records to a Sink.

Subscriber-only (no request-path proxy; the observe tier). The real interface is
`ingest(source_id, event)` over normalized events; `ingest_batch`/`run` are compat helpers
that push native vLLM batches through the vLLM Source first.

  1. Residual dedup (number one). A StoredBlock whose hash is currently RESIDENT on another
     source is recompute a cross-instance cache would have avoided. (Blocks a source served
     from its OWN cache are never stored, so stores are already net of local hits.)
  2. Eviction waste (number two). A StoredBlock whose hash was previously evicted from the
     SAME source is eviction-driven recompute.
  Plus day-one-schema records to the Sink: stack provenance, salted block ids, lifecycle.

Correctness assumption (operationally enforced): sources share the hash seed + algo, so the
same prefix hashes identically across them; else cross-instance overlap is silently zero (a
live minute-one check, not agent logic).
"""

import hashlib
from collections import OrderedDict

# Record keys that would carry KV *content* rather than metadata. Enforced by
# assert_metadata_only: content is never retained, serialized, or transmitted (token ids
# in vLLM's event stream are dropped at decode; KV tensors never leave the GPU).
_FORBIDDEN_CONTENT_KEYS = frozenset({"token_ids", "tokens", "text", "content", "prompt"})


def salted_id(raw_hash, domain, salt):
    """Salted, per-domain block id. Raw content-hash ids never leave their domain (a
    content hash is a content fingerprint); the salt makes them non-reversible across
    deployments. Deterministic within a (salt, domain) so cross-instance overlap is still
    detectable after salting."""
    h = hashlib.sha256()
    h.update(salt.encode() if isinstance(salt, str) else salt)
    h.update(b"\x00")
    h.update(domain.encode())
    h.update(b"\x00")
    h.update(repr(raw_hash).encode())
    return h.hexdigest()[:16]


def assert_metadata_only(record):
    """Wire-enforceable guard: a record carries only metadata (counts, salted ids,
    timestamps, lifecycle), never KV contents. Raises on a forbidden key or on a value
    that looks like a raw token sequence (a long list of ints)."""
    for k, v in record.items():
        if k in _FORBIDDEN_CONTENT_KEYS:
            raise ValueError(f"record leaks KV content via key {k!r}")
        if isinstance(v, (list, tuple)) and len(v) >= 4 and all(isinstance(x, int) for x in v):
            raise ValueError(f"record field {k!r} looks like raw token ids ({len(v)} ints)")
    return record


class TelemetryAgent:
    def __init__(self, provenance, salt, block_size, salt_domain="default",
                 record_cap=None, sink=None, evicted_cap=None):
        # provenance: stack fields stamped into every record (stack provenance).
        # Must include event_schema_version so the corpus survives a vLLM bump.
        self.provenance = dict(provenance)
        self.salt = salt
        self.salt_domain = salt_domain
        self.block_size = block_size
        # source_id -> set of currently-resident block hashes. Naturally bounded: it tracks
        # each engine's real cache (grows on store, shrinks on remove/clear), so it cannot
        # exceed the engine's block count. Raw in-memory only; records carry salted_id.
        self._resident = {}
        # source_id -> ORDERED set of hashes evicted from THIS source and not yet re-stored.
        # A later StoredBlock of one is eviction-driven recompute (number two). This is the
        # one unbounded structure -> capped + aged-out (evicted_cap): oldest dropped first,
        # since a block re-referenced only after aging out is not near-term eviction waste.
        # A dropped entry that IS re-stored later is a missed count -> surfaced as
        # evicted_aged_out so the (small, stale) undercount is visible, not silent.
        self._evicted = {}
        self.evicted_cap = evicted_cap
        self.evicted_aged_out = 0
        self.residual_tokens = 0            # number one: cross-instance-avoidable recompute
        self.eviction_recompute_tokens = 0  # number two: within-instance eviction recompute
        # Records go to a Sink. Default: a bounded in-RAM MemorySink so a long run can't OOM
        # (a real deployment injects a streaming sink); counts stay exact regardless.
        from .sinks import MemorySink
        self._sink = sink if sink is not None else MemorySink(cap=record_cap)
        self.events_ingested = 0
        # peak (resident+evicted) raw-hash entries across sources: the catalog memory driver.
        self.catalog_peak = 0

    # compat surface: records/record_counts read through whatever sink is attached
    @property
    def records(self):
        return getattr(self._sink, "records", [])

    @property
    def record_counts(self):
        return getattr(self._sink, "counts", {})

    # --- resident-set maintenance ---------------------------------------------
    def _resident_elsewhere(self, source_id, h):
        for other, held in self._resident.items():
            if other != source_id and h in held:
                return other
        return None

    def _emit(self, record):
        self._sink.emit(assert_metadata_only(record))

    def _evict_add(self, evicted, h):
        # ordered-set add: most-recently-evicted at the end; drop the oldest past the cap.
        evicted.pop(h, None)
        evicted[h] = None
        if self.evicted_cap is not None and len(evicted) > self.evicted_cap:
            evicted.popitem(last=False)  # oldest out; a re-store past here is a missed count
            self.evicted_aged_out += 1

    def ingest(self, source_id, event):
        """Ingest ONE normalized event (event_model) from `source_id`. Events MUST arrive in
        global timestamp order across sources so 'resident elsewhere' is as-of store time.
        This is the real Compute interface; it consumes only the normalized model."""
        self.events_ingested += 1
        held = self._resident.setdefault(source_id, set())
        evicted = self._evicted.setdefault(source_id, OrderedDict())
        kind = type(event).__name__
        if kind == "StoredBlock":
            h = event.block_hash
            elsewhere = self._resident_elsewhere(source_id, h)
            held.add(h)
            if h in evicted:
                # number two: evicted from THIS source, now recomputed and re-stored.
                evicted.pop(h, None)
                self.eviction_recompute_tokens += self.block_size
                self._emit({**self.provenance, "kind": "eviction_recompute",
                            "instance_id": source_id,
                            "block_id": salted_id(h, self.salt_domain, self.salt),
                            "n_tokens": self.block_size, "medium": event.medium,
                            "at_ms": event.ts})
            if elsewhere is not None:
                # number one: resident on another source -> cross-instance-avoidable.
                self.residual_tokens += self.block_size
                self._emit({**self.provenance, "kind": "residual_block",
                            "instance_id": source_id, "also_on_instance": elsewhere,
                            "block_id": salted_id(h, self.salt_domain, self.salt),
                            "n_tokens": self.block_size, "medium": event.medium,
                            "at_ms": event.ts})
        elif kind == "RemovedBlock":
            h = event.block_hash
            held.discard(h)
            self._evict_add(evicted, h)  # dropped; a later re-store is eviction recompute
            self._emit(self._lifecycle_record(source_id, h, event.medium, event.ts, "evict"))
        elif kind == "ClearedAll":
            # sorted, not set-iteration order, so the emitted record order is deterministic
            # and language-neutral (a Rust port reproduces it exactly).
            for h in sorted(held):
                self._evict_add(evicted, h)
                self._emit(self._lifecycle_record(source_id, h, event.medium, event.ts,
                                                  "clear"))
            held.clear()
        tracked = (sum(len(s) for s in self._resident.values())
                   + sum(len(s) for s in self._evicted.values()))
        if tracked > self.catalog_peak:
            self.catalog_peak = tracked

    def ingest_batch(self, source_id, batch):
        """Compat: push a native vLLM KVEventBatch through the vLLM Source and ingest the
        normalized events. The real interface is `ingest(source_id, event)`."""
        from .sources import normalize_vllm_batch
        for sid, ev in normalize_vllm_batch(source_id, batch):
            self.ingest(sid, ev)

    def catalog_stats(self):
        """The agent's own memory footprint drivers, for the resource-envelope claim: how
        many raw block-hash entries it holds. Peak sizes the configurable catalog cap."""
        resident = sum(len(s) for s in self._resident.values())
        evicted = sum(len(s) for s in self._evicted.values())
        return {
            "catalog_peak_entries": self.catalog_peak,
            "resident_entries": resident,
            "evicted_entries": evicted,
            "tracked_current": resident + evicted,
            "evicted_aged_out": self.evicted_aged_out,
            "record_counts": dict(self.record_counts),
            "records_sampled": len(self.records),
            "events_ingested": self.events_ingested,
        }

    def _lifecycle_record(self, source_id, h, medium, ts, reason):
        # Retention / lifecycle audit trail. Records posture, never a destruction guarantee.
        return {**self.provenance, "kind": "lifecycle", "reason": reason,
                "instance_id": source_id,
                "block_id": salted_id(h, self.salt_domain, self.salt),
                "tier": medium, "at_ms": ts}

    # --- driver ----------------------------------------------------------------
    def run(self, streams):
        """streams: {source_id: [KVEventBatch, ...]}. Merges by ts and ingests in global
        order. Returns total residual_tokens."""
        merged = sorted(
            ((b.ts, inst, b) for inst, batches in streams.items() for b in batches),
            key=lambda t: t[0],
        )
        for _ts, inst, batch in merged:
            self.ingest_batch(inst, batch)
        return self.residual_tokens
