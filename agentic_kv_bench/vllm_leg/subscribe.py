"""Live ZMQ subscriber that feeds decoded vLLM KV events into the telemetry agent, plus
the minute-one sanity checks (docs/vllm-leg-design.md §6). zmq/msgspec are box-only, so
they are imported lazily; the sanity-check logic is pure and unit-tested locally.

Wire (v0.11.0): SUB socket, 3-part multipart (topic, seq:8-byte-big-endian, msgpack).
`msgspec.msgpack.decode(payload)` with no type yields the raw array form (array_like+tag),
which is exactly what `decode_batch` consumes."""

from .kv_events import decode_batch
from .mock_stream import wire_fingerprint


def check_schema(live_batch_arr, known_fingerprints=None):
    """Minute-one schema gate (§6). The hard gate is that `decode_batch` accepts the live
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
    """PYTHONHASHSEED cross-instance check (§6): feed BOTH instances one identical probe
    prefix, collect the BlockStored hashes each emits, and confirm they MATCH. If they
    differ, NONE_HASH differs across instances, prefix-chained hashes diverge, and
    cross-instance overlap silently reads zero. Returns True on agreement.

    stored_hashes_by_instance: {instance_id: [hash, ...]} for the SAME probe prefix."""
    sets = list(stored_hashes_by_instance.values())
    if len(sets) < 2:
        return True
    first = list(sets[0])
    return all(list(s) == first for s in sets[1:])


def subscribe(endpoints, agent, golden_fingerprint=None, max_batches=None,
              duration_s=None):
    """Subscribe to each instance's ZMQ endpoint and feed decoded batches to `agent` in
    arrival order. endpoints: {instance_id: "tcp://host:port"}. Box-only (needs zmq +
    msgspec). The first batch per instance is schema-checked against golden_fingerprint.
    Stops at max_batches, after duration_s wall-seconds, or on SIGINT — whichever first;
    the caller dumps agent state after this returns (so a timed measurement window works)."""
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
    deadline = None if duration_s is None else time.monotonic() + duration_s
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
                _topic, _seq, payload = s.recv_multipart()
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
    return agent


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
    p.add_argument("--out", required=True, help="output records JSON path")
    a = p.parse_args(argv)

    endpoints = {}
    for pair in a.endpoints.split(","):
        k, v = pair.split("=", 1)
        endpoints[int(k)] = v
    with open(a.provenance) as f:
        provenance = json.load(f)
    agent = TelemetryAgent(provenance, salt=a.salt, block_size=a.block_size)
    subscribe(endpoints, agent, golden_fingerprint=GOLDEN_FINGERPRINT,
              max_batches=a.max_batches, duration_s=a.duration)
    with open(a.out, "w") as f:
        json.dump({"residual_tokens": agent.residual_tokens,
                   "records": agent.records}, f, indent=1)
    print(f"residual_tokens={agent.residual_tokens} records={len(agent.records)} -> {a.out}")


if __name__ == "__main__":
    main()
