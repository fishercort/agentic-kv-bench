"""Live ZMQ subscriber that feeds decoded vLLM KV events into the telemetry agent, plus
the minute-one sanity checks. zmq/msgspec are box-only, so
they are imported lazily; the sanity-check logic is pure and unit-tested locally.

Wire (v0.11.0): SUB socket, 3-part multipart (topic, seq:8-byte-big-endian, msgpack).
`msgspec.msgpack.decode(payload)` with no type yields the raw array form (array_like+tag),
which is exactly what `decode_batch` consumes."""

from .kv_events import decode_batch
from .mock_stream import wire_fingerprint


def check_schema(live_batch_arr, known_fingerprints=None):
    """Minute-one schema gate. The hard gate is that `decode_batch` accepts the live
    batch — it raises SchemaDriftError on an unknown tag or wrong arity, which IS the
    abort. Returns (fingerprint, is_known): a fingerprint not among the pinned golden
    shapes is not necessarily drift (omit_defaults varies the optional `medium`), so it is
    a logged diff, not a raise. `known_fingerprints`: iterable of pinned shapes to compare
    against."""
    decode_batch(live_batch_arr)  # raises SchemaDriftError on real drift
    fp = wire_fingerprint(live_batch_arr)
    is_known = known_fingerprints is None or fp in set(known_fingerprints)
    return fp, is_known


def seed_agreement(stored_hashes_by_instance):
    """PYTHONHASHSEED cross-instance check: feed BOTH instances one identical probe
    prefix, collect the BlockStored hashes each emits, and confirm they MATCH. If they
    differ, NONE_HASH differs across instances, prefix-chained hashes diverge, and
    cross-instance overlap silently reads zero. Returns True on agreement.

    stored_hashes_by_instance: {instance_id: [hash, ...]} for the SAME probe prefix."""
    sets = list(stored_hashes_by_instance.values())
    if len(sets) < 2:
        return True
    first = list(sets[0])
    return all(list(s) == first for s in sets[1:])


def seq_gap(prev, cur):
    """Missing-message count between two consecutive ZMQ publisher sequence numbers. vLLM
    stamps each multipart message with a monotonic seq (8-byte big-endian); a jump means
    the SUB socket dropped events under load — which would silently UNDERCOUNT residual /
    eviction. Returns messages lost between prev and cur (0 if in-order or a reset)."""
    if prev is None or cur <= prev:
        return 0
    return cur - prev - 1


def subscribe(endpoints, agent, golden_fingerprint=None, max_batches=None,
              duration_s=None):
    """Subscribe to each instance's ZMQ endpoint and feed decoded batches to `agent` in
    arrival order. endpoints: {instance_id: "tcp://host:port"}. Box-only (needs zmq +
    msgspec). The first batch per instance is schema-checked against golden_fingerprint.
    Stops at max_batches, after duration_s wall-seconds, or on SIGINT — whichever first;
    the caller dumps agent state after this returns (so a timed measurement window works).
    Returns an ingest-meta dict: messages seen, dropped-message count per instance (from
    seq gaps), and wall seconds — the lag/drop evidence for the resource envelope."""
    import time

    import msgspec  # box-only
    import zmq  # box-only

    ctx = zmq.Context.instance()
    poller = zmq.Poller()
    socks = {}
    for inst, ep in endpoints.items():
        s = ctx.socket(zmq.SUB)
        s.connect(ep)
        s.setsockopt(zmq.SUBSCRIBE, b"")  # all topics
        poller.register(s, zmq.POLLIN)
        socks[s] = inst
    checked = set()
    seen = 0
    last_seq = {}      # instance -> last seq seen
    dropped = {}       # instance -> total messages dropped (seq gaps)
    started = time.monotonic()
    deadline = None if duration_s is None else started + duration_s
    try:
        while max_batches is None or seen < max_batches:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ready = poller.poll(timeout=min(500.0, remaining * 1000))
            else:
                ready = poller.poll(timeout=500.0)
            for s, _ in ready:
                inst = socks[s]
                _topic, seq_bytes, payload = s.recv_multipart()
                seq = int.from_bytes(seq_bytes, "big")
                dropped[inst] = dropped.get(inst, 0) + seq_gap(last_seq.get(inst), seq)
                last_seq[inst] = seq
                arr = msgspec.msgpack.decode(payload)
                if golden_fingerprint is not None and inst not in checked:
                    known = golden_fingerprint.get(inst) if isinstance(
                        golden_fingerprint, dict) else golden_fingerprint
                    check_schema(arr, known)  # raises SchemaDriftError on real drift
                    checked.add(inst)
                agent.ingest_batch(inst, decode_batch(arr))
                seen += 1
    except KeyboardInterrupt:
        pass  # clean stop -> caller still dumps agent state
    finally:
        for s in socks:
            s.close(0)
    return {"messages_seen": seen, "dropped_by_instance": dropped,
            "dropped_total": sum(dropped.values()),
            "wall_seconds": time.monotonic() - started}


def _show_payload():
    """Print exactly what the agent retains and emits, on the version-pinned mock stream,
    and prove it is metadata-only. Standalone (no live vLLM). This is the trust check a
    security reviewer runs: here is everything the process holds and would transmit."""
    import json

    from .agent import TelemetryAgent, assert_metadata_only
    from .kv_events import EVENT_SCHEMA_VERSION
    from .mock_stream import GOLDEN, golden_batches

    prov = {"gpu": "<example>", "vllm_version": "0.11.0",
            "event_schema_version": EVENT_SCHEMA_VERSION, "block_size": GOLDEN["block_size"]}
    agent = TelemetryAgent(prov, salt="example-deployment-salt",
                           block_size=GOLDEN["block_size"])

    print("=== INGRESS: a decoded vLLM BlockStored event, as the agent holds it ===")
    ev = golden_batches(0)[0].events[0]
    print(f"  {type(ev).__name__}: block_hashes={ev.block_hashes} block_size={ev.block_size} "
          f"medium={ev.medium} token_ids={ev.token_ids}")
    print("  ^ token_ids is None: token content is dropped at the decode boundary, never held.")

    for inst in GOLDEN["streams"]:
        for batch in golden_batches(inst):
            agent.ingest_batch(inst, batch)

    print("\n=== EGRESS: every record the agent retains / would transmit ===")
    for r in agent.records:
        assert_metadata_only(r)  # raises if any content slipped in
        print("  " + json.dumps(r))
    print(f"\n{len(agent.records)} records, all passed assert_metadata_only.")
    print("Contents: salted block ids, token COUNTS, timestamps, lifecycle, stack provenance.")
    print("Never present: token ids, text, or KV tensors (KV tensors never leave the GPU).")


def main(argv=None):
    """CLI: subscribe to both instances, run the agent, dump records + residual to --out.
    Box-only (needs zmq/msgspec via `uv sync --extra box`). `--show-payload` runs standalone
    (no vLLM) to verify the metadata-only guarantee."""
    import argparse
    import json

    from .agent import TelemetryAgent
    from .mock_stream import GOLDEN_FINGERPRINT

    p = argparse.ArgumentParser(description="vLLM-leg telemetry agent (live subscriber)")
    p.add_argument("--endpoints",
                   help='e.g. "0=tcp://localhost:5557,1=tcp://localhost:5558"')
    p.add_argument("--provenance", help="path to stack-provenance JSON")
    p.add_argument("--salt", help="deployment salt for block-id hashing")
    p.add_argument("--block-size", type=int)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--duration", type=float, default=None,
                   help="stop after N wall-seconds and dump (the measurement window); "
                        "Ctrl-C also stops cleanly and dumps")
    p.add_argument("--record-cap", type=int, default=100_000,
                   help="max per-event records held in RAM (counts stay exact); a long "
                        "run would otherwise OOM. A real deployment streams to a sink.")
    p.add_argument("--evicted-cap", type=int, default=None,
                   help="bound the per-source evicted-hash catalog (age out oldest past this; "
                        "sized from the measured catalog peak). Unset = unbounded.")
    p.add_argument("--out", help="output records JSON path")
    p.add_argument("--show-payload", action="store_true",
                   help="print exactly what the agent retains and emits (metadata only), on "
                        "the pinned mock stream, and exit -- verify no KV content leaves the "
                        "host. Needs no live vLLM.")
    a = p.parse_args(argv)

    if a.show_payload:
        _show_payload()
        return
    missing = [n for n in ("endpoints", "provenance", "salt", "block_size", "out")
               if getattr(a, n) in (None,)]
    if missing:
        p.error("required unless --show-payload: " + ", ".join("--" + m.replace("_", "-")
                                                               for m in missing))

    endpoints = {}
    for pair in a.endpoints.split(","):
        k, v = pair.split("=", 1)
        endpoints[int(k)] = v
    with open(a.provenance) as f:
        provenance = json.load(f)
    agent = TelemetryAgent(provenance, salt=a.salt, block_size=a.block_size,
                           record_cap=a.record_cap, evicted_cap=a.evicted_cap)
    meta = subscribe(endpoints, agent, golden_fingerprint=GOLDEN_FINGERPRINT,
                     max_batches=a.max_batches, duration_s=a.duration)

    import resource
    import sys
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is KiB on Linux (the box), bytes on macOS; record raw + platform.
    envelope = {
        "peak_rss_ru_maxrss": ru.ru_maxrss, "ru_maxrss_unit":
            "KiB" if sys.platform.startswith("linux") else "bytes",
        "cpu_seconds": round(ru.ru_utime + ru.ru_stime, 2),
        "wall_seconds": round(meta["wall_seconds"], 2),
        "avg_cores": round((ru.ru_utime + ru.ru_stime) / max(meta["wall_seconds"], 1e-9), 3),
        "messages_seen": meta["messages_seen"],
        "dropped_total": meta["dropped_total"],
        "dropped_by_instance": meta["dropped_by_instance"],
        **agent.catalog_stats(),
    }
    with open(a.out, "w") as f:
        json.dump({"residual_tokens": agent.residual_tokens,
                   "eviction_recompute_tokens": agent.eviction_recompute_tokens,
                   "resource_envelope": envelope,
                   "records": agent.records}, f, indent=1)
    print(f"residual={agent.residual_tokens} eviction={agent.eviction_recompute_tokens} "
          f"msgs={meta['messages_seen']} dropped={meta['dropped_total']} "
          f"catalog_peak={agent.catalog_peak} avg_cores={envelope['avg_cores']} -> {a.out}")
    if meta["dropped_total"]:
        print(f"WARNING: {meta['dropped_total']} messages dropped (seq gaps) -> numbers "
              "UNDERCOUNT; lower --speedup or raise vLLM kv-events hwm and re-run.")


if __name__ == "__main__":
    main()
