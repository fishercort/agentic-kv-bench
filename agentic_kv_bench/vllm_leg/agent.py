"""The telemetry agent: the shipping deliverable, coded and tested with zero GPU time.

Subscriber-only (no request-path proxy; L2 §0.1 observe tier). It consumes decoded
KV-event batches from N vLLM instances and produces two things:

  1. Residual dedup (headline number one), event-driven. vLLM emits `BlockStored` only
     for blocks it actually (re)computed and cached — a block served from the instance's
     OWN prefix cache (counted in Prometheus `num_local_cached_tokens`) is never stored.
     So the set of stored block hashes is ALREADY net of local APC hits. A stored block
     whose hash is currently RESIDENT on another instance is recompute a cross-instance
     cache would have avoided: that is the residual. residual_tokens = residual_blocks *
     block_size. `num_local_cached_tokens` is carried for the cross-check and dollarization,
     not re-subtracted (it is baked into what does/doesn't get stored). See
     docs/vllm-leg-design.md §2, §5.

  2. Day-one schema records (docs §1) — the impossible-to-retrofit fields, emitted ONCE
     here: stack provenance, salted hash domains, metadata-only-at-the-wire, retention /
     lifecycle events, per-tenant attribution, counterfactual forward window.

Correctness assumption (runbook-enforced, §6): both instances share PYTHONHASHSEED + hash
algo, so the same prefix hashes identically across instances. Without that, cross-instance
overlap is silently zero; the agent cannot detect it from events alone, so it is a live
minute-one check, not agent logic.
"""

import hashlib

# Record keys that would carry KV *content* rather than metadata. Enforced by
# assert_metadata_only: contents are never read, serialized, or transmitted (§1).
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
    def __init__(self, provenance, salt, block_size, salt_domain="default"):
        # provenance: stack fields stamped into every record (§1 stack provenance).
        # Must include event_schema_version so the corpus survives a vLLM bump.
        self.provenance = dict(provenance)
        self.salt = salt
        self.salt_domain = salt_domain
        self.block_size = block_size
        # instance_id -> set of currently-resident block hashes (raw, in-memory only;
        # never emitted raw — records carry salted_id).
        self._resident = {}
        # instance_id -> set of hashes evicted from THIS instance and not yet re-stored.
        # A later BlockStored of one of these is eviction-driven recompute (number two):
        # the block was needed again after being dropped. This is what LRU costs under
        # pressure that a larger/oracle cache would have avoided.
        self._evicted = {}
        self.residual_tokens = 0          # number one: cross-instance-avoidable recompute
        self.eviction_recompute_tokens = 0  # number two: within-instance eviction-driven recompute
        self.records = []  # emitted day-one-schema records

    # --- resident-set maintenance ---------------------------------------------
    def _resident_elsewhere(self, instance_id, h):
        for other, held in self._resident.items():
            if other != instance_id and h in held:
                return other
        return None

    def ingest_batch(self, instance_id, batch):
        """Ingest one decoded KVEventBatch from `instance_id`. Batches MUST be fed in
        global timestamp order across instances so 'resident elsewhere' is as-of store
        time. Returns the records emitted for this batch."""
        held = self._resident.setdefault(instance_id, set())
        evicted = self._evicted.setdefault(instance_id, set())
        emitted = []
        for ev in batch.events:
            tag = type(ev).__name__
            if tag == "BlockStored":
                emitted += self._on_stored(instance_id, held, evicted, ev, batch.ts)
            elif tag == "BlockRemoved":
                for h in ev.block_hashes:
                    held.discard(h)
                    evicted.add(h)  # dropped; a later re-store is eviction recompute
                    emitted.append(self._lifecycle_record(instance_id, h, ev.medium,
                                                          batch.ts, "evict"))
            elif tag == "AllBlocksCleared":
                for h in list(held):
                    evicted.add(h)
                    emitted.append(self._lifecycle_record(instance_id, h, ev.medium,
                                                          batch.ts, "clear"))
                held.clear()
        self.records += emitted
        return emitted

    def _on_stored(self, instance_id, held, evicted, ev, ts):
        emitted = []
        for h in ev.block_hashes:
            elsewhere = self._resident_elsewhere(instance_id, h)
            held.add(h)
            if h in evicted:
                # number two: this block was evicted from THIS instance and is now being
                # recomputed and re-stored -> eviction-driven recompute (LRU's cost under
                # pressure). It is no longer in the evicted set now that it is back.
                evicted.discard(h)
                self.eviction_recompute_tokens += self.block_size
                emitted.append(assert_metadata_only({
                    **self.provenance,
                    "kind": "eviction_recompute",
                    "instance_id": instance_id,
                    "block_id": salted_id(h, self.salt_domain, self.salt),
                    "n_tokens": self.block_size,
                    "medium": ev.medium,
                    "at_ms": ts,
                }))
            if elsewhere is not None:
                # number one: this block is resident elsewhere -> cross-instance-avoidable.
                self.residual_tokens += self.block_size
                emitted.append(assert_metadata_only({
                    **self.provenance,
                    "kind": "residual_block",
                    "instance_id": instance_id,
                    "also_on_instance": elsewhere,
                    "block_id": salted_id(h, self.salt_domain, self.salt),
                    "n_tokens": self.block_size,
                    "medium": ev.medium,
                    "at_ms": ts,
                }))
        return emitted

    def _lifecycle_record(self, instance_id, h, medium, ts, reason):
        # Retention / lifecycle audit trail (§1). Records posture, never a destruction
        # guarantee. counterfactual_window filled by the band analysis downstream.
        return assert_metadata_only({
            **self.provenance,
            "kind": "lifecycle",
            "reason": reason,
            "instance_id": instance_id,
            "block_id": salted_id(h, self.salt_domain, self.salt),
            "tier": medium,
            "at_ms": ts,
        })

    # --- driver ----------------------------------------------------------------
    def run(self, streams):
        """streams: {instance_id: [KVEventBatch, ...]}. Merges by ts and ingests in
        global order. Returns total residual_tokens."""
        merged = sorted(
            ((b.ts, inst, b) for inst, batches in streams.items() for b in batches),
            key=lambda t: t[0],
        )
        for _ts, inst, batch in merged:
            self.ingest_batch(inst, batch)
        return self.residual_tokens
