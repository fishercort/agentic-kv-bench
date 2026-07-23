"""Replay driver: shard the synthesized corpus across the fleet and send each session's
requests to vLLM's OpenAI-compatible /v1/completions, pacing by the trace's arrival times
(docs/vllm-leg-design.md §3, §4). httpx is box-only and imported lazily; sharding and
pacing are pure and unit-tested locally.

vLLM /v1/completions accepts `prompt` as a list of token ids directly, so the synthesized
token ids (declared-prefix mode) go on the wire without a tokenizer round-trip."""

from .replay import synth_session


def shard_traces(traces, n_instances):
    """Assign traces to instances deterministically by trace id (stable across runs, no
    RNG). Returns {instance_id: [trace, ...]}. Session affinity is preserved (a whole
    session lands on one instance) so within-session prefix reuse stays local; cross-
    session sharing is what the modeled prefix creates across instances."""
    shards = {i: [] for i in range(n_instances)}
    for t in sorted(traces, key=lambda t: t["id"]):
        # sum of id chars: deterministic, no hashing-seed dependence
        bucket = sum(ord(c) for c in t["id"]) % n_instances
        shards[bucket].append(t)
    return shards


def pace(requests, speedup=1.0):
    """Yield (delay_s_before_this_request, request) from a session's requests, pacing by
    the trace `t` deltas. speedup>1 compresses wall-clock (speedup=inf via 0.0 => asap)."""
    prev = None
    for r in requests:
        if prev is None or speedup == float("inf"):
            delay = 0.0
        else:
            delay = max(0.0, (r["t"] - prev) / speedup)
        prev = r["t"]
        yield delay, r


def build_workload(traces, n_instances, shared_prefix_blocks, block_size, vocab_size):
    """Full offline prep: shard, then synthesize each session (declared-prefix mode).
    Returns {instance_id: [session, ...]} where a session is the synth_session output.
    Everything network-free, so the exact bytes sent are decided before the meter."""
    shards = shard_traces(traces, n_instances)
    return {
        inst: [synth_session(t, shared_prefix_blocks, block_size, vocab_size)
               for t in shard]
        for inst, shard in shards.items()
    }


def send_completion(base_url, token_ids, max_tokens, model, timeout=60.0):
    """POST one prefill+decode to vLLM. Box-only (httpx). Returns the response JSON, whose
    `usage.prompt_tokens_details.cached_tokens` (the OpenAI-surfaced name for the internal
    `num_local_cached_tokens`) gives the local-APC cross-check (§2)."""
    import httpx  # box-only

    resp = httpx.post(
        f"{base_url}/v1/completions",
        json={"model": model, "prompt": token_ids, "max_tokens": max_tokens,
              "temperature": 0.0},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _load_corpus(corpus_dir):
    import glob
    import json
    traces = []
    for f in sorted(glob.glob(f"{corpus_dir}/trace_*.json")):
        with open(f) as fh:
            traces.append(json.load(fh))
    return traces


def trace_footprint(trace):
    """Unique KV footprint of a session in tokens: distinct block hashes * corpus
    block_size. This is what sits resident and drives cache pressure."""
    bs = trace.get("block_size", 64)
    distinct = set()
    for r in trace["requests"]:
        distinct.update(r.get("hash_ids", []))
    return len(distinct) * bs


def select_by_footprint(traces, footprint_cap):
    """Split traces into (kept, excluded) at the footprint cap. The excluded deep tail is
    returned so the caller can LOG it (no silent truncation) — number two is measured on
    the kept band and the exclusion is reported."""
    if footprint_cap is None:
        return list(traces), []
    kept, excluded = [], []
    for t in traces:
        (kept if trace_footprint(t) <= footprint_cap else excluded).append(t)
    return kept, excluded


def main(argv=None):
    """CLI: shard + synthesize the corpus and replay each shard into its instance,
    sessions launched concurrently and paced by arrival time. Box-only (needs httpx)."""
    import argparse
    import threading
    import time

    p = argparse.ArgumentParser(description="vLLM-leg replay driver (declared-prefix mode)")
    p.add_argument("--corpus", required=True, help="corpus dir of trace_*.json")
    p.add_argument("--instances", required=True,
                   help='e.g. "0=http://localhost:8001,1=http://localhost:8002"')
    p.add_argument("--shared-prefix-blocks", type=int, required=True,
                   help="S: modeled shared system+tool prefix (see replay.RESIDUAL_LABEL)")
    p.add_argument("--block-size", type=int, required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--speedup", type=float, default=1.0, help="inf = asap")
    p.add_argument("--max-concurrent", type=int, default=None,
                   help="cap in-flight sessions (sets concurrent working set = the band); "
                        "unset = all at once (only for tiny smoke runs)")
    p.add_argument("--footprint-cap", type=int, default=None,
                   help="replay only sessions whose unique KV footprint (distinct blocks * "
                        "corpus block_size) <= this; the excluded deep tail is logged, not hidden")
    p.add_argument("--served-model", required=True,
                   help="the model name vLLM actually serves (e.g. "
                        "NousResearch/Meta-Llama-3.1-8B-Instruct); sent on every request. "
                        "Distinct from the trace's model, which only keys the shared prefix.")
    a = p.parse_args(argv)

    base = {}
    for pair in a.instances.split(","):
        k, v = pair.split("=", 1)
        base[int(k)] = v
    traces = _load_corpus(a.corpus)
    kept, excluded = select_by_footprint(traces, a.footprint_cap)
    if excluded:
        # no silent truncation: the deep tail is reported, not hidden (§3, provenance rule)
        print(f"footprint-cap {a.footprint_cap}: replaying {len(kept)} sessions, "
              f"EXCLUDED {len(excluded)} deep-tail sessions (reported in the leg's scope)")
    workload = build_workload(kept, len(base), a.shared_prefix_blocks,
                              a.block_size, a.vocab_size)

    # concurrency cap = the band control: at most K sessions in flight -> concurrent
    # working set = K * mean_footprint, which sets cap/demand (see runbook band table).
    gate = threading.Semaphore(a.max_concurrent) if a.max_concurrent else None

    def run_session(url, sess):
        if gate:
            gate.acquire()
        try:
            for delay, r in pace(sess, a.speedup):
                if delay:
                    time.sleep(delay)
                # served model on the wire (Llama); r["model"] (Claude) only keyed synth
                send_completion(url, r["token_ids"], max(1, r["out"]), a.served_model)
        finally:
            if gate:
                gate.release()

    threads = []
    for inst, sessions in workload.items():
        for sess in sessions:
            t = threading.Thread(target=run_session, args=(base[inst], sess), daemon=True)
            threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    n_sess = sum(len(s) for s in workload.values())
    print(f"replayed {n_sess} sessions across {len(base)} instances "
          f"(max_concurrent={a.max_concurrent})")


if __name__ == "__main__":
    main()
