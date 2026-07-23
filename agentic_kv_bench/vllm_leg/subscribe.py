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


def main(argv=None):
    """CLI: subscribe to both instances, run the agent, dump records + residual to --out.
    Box-only (needs zmq/msgspec via `uv sync --extra box`)."""
    import argparse
    import json

    from .agent import TelemetryAgent
    from .mock_stream import GOLDEN_FINGERPRINT

    p = argparse.ArgumentParser(description="vLLM-leg telemetry agent (live subscriber)")
    p.add_argument("--endpoints", required=True,
                   help='e.g. "0=tcp://localhost:5557,1=tcp://localhost:5558"')
    p.add_argument("--provenance", required=True, help="path to stack-provenance JSON")
    p.add_argument("--salt", required=True, help="deployment salt for block-id hashing")
    p.add_argument("--block-size", type=int, required=True)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--duration", type=float, default=None,
                   help="stop after N wall-seconds and dump (the measurement window); "
                        "Ctrl-C also stops cleanly and dumps")
    p.add_argument("--record-cap", type=int, default=100_000,
                   help="max per-event records held in RAM (counts stay exact); a long "
                        "run would otherwise OOM. A real deployment streams to a sink.")
    p.add_argument("--out", required=True, help="output records JSON path")
    a = p.parse_args(argv)

    endpoints = {}
    for pair in a.endpoints.split(","):
        k, v = pair.split("=", 1)
        endpoints[int(k)] = v
    with open(a.provenance) as f:
        provenance = json.load(f)
    agent = TelemetryAgent(provenance, salt=a.salt, block_size=a.block_size,
                           record_cap=a.record_cap)
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
