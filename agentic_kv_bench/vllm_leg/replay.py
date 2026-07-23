"""Declared-prefix synthesis mode for the replayer.

The corpus is `hash_id_scope: local` for all 739 traces (verified) and every trace
numbers blocks from 1, so it preserves WITHIN-session structure but discards CROSS-session
identity. Residual dedup is definitionally cross-session, so it cannot be read from the
trace hash ids: honest per-trace namespacing gives zero, and reusing raw ids across traces
fabricates the entire number.

Real Claude Code sessions DO share content: the system-prompt preamble + tool schema,
identical across sessions on the same model. This module MODELS that sharing explicitly:
the first `S` blocks of every session are a shared, model-keyed prefix (identical tokens
across all traces of a model, so vLLM prefix-caches them and the telemetry agent detects
the cross-instance overlap); everything past block S is per-trace namespaced, so within-
session structure is preserved and NO false cross-session sharing is created.

S is not assumed: `measured_shared_prefix_blocks` reports the empirical distribution from
the corpus's real system_tokens/tool_tokens so the sweep's x-axis is grounded. The
caveat below travels with every number this produces.
"""

import math

from .synth import synth_block_tokens, vllm_chain_hashes

# The label that travels with headline number one everywhere it appears (report line,
# results JSON, RFC exhibit) — the caveat is part of the number, same rule as the
# TTL-strawman label.
RESIDUAL_LABEL = (
    "modeled cross-session sharing (declared shared prefix; "
    "corpus does not preserve cross-session identity)"
)


def _block_key(hash_id, trace_id, model, shared_prefix_blocks):
    """Map a trace block to its synthesis namespace. Leading S blocks -> SHARED (keyed by
    model only, so identical across every trace of that model). Rest -> LOCAL (keyed by
    trace, so within-session sharing is preserved, cross-session is not)."""
    if hash_id <= shared_prefix_blocks:
        return ("SHARED", model, hash_id)
    return ("LOCAL", trace_id, hash_id)


def synth_request_tokens(hash_ids, trace_id, model, shared_prefix_blocks,
                         block_size, vocab_size):
    """Token ids for one request: concatenation of the namespaced synthesis of its
    block-hash chain."""
    toks = []
    for h in hash_ids:
        key = _block_key(h, trace_id, model, shared_prefix_blocks)
        toks.extend(synth_block_tokens(key, block_size, vocab_size))
    return toks


def synth_session(trace, shared_prefix_blocks, block_size, vocab_size):
    """Turn one corpus trace into a replayable session: a list of requests, each with its
    arrival time, output length, and synthesized input token ids. Structure is the real
    trace's; content is synthesized; the shared prefix is modeled (see RESIDUAL_LABEL)."""
    trace_id = trace["id"]
    model = trace["models"][0]
    out = []
    for r in trace["requests"]:
        out.append({
            "t": r["t"],
            "out": r.get("out", 0),
            "model": model,
            "token_ids": synth_request_tokens(
                r.get("hash_ids", []), trace_id, model,
                shared_prefix_blocks, block_size, vocab_size),
        })
    return out


def measured_shared_prefix_blocks(traces, block_size):
    """Empirical S distribution grounding the sweep: system+tool prefix in blocks,
    per trace, at the given block_size. Returns sorted list; caller reports percentiles /
    where mass concentrates. S is measured, not assumed."""
    vals = []
    for t in traces:
        prefix_tokens = t.get("system_tokens", 0) + t.get("tool_tokens", 0)
        vals.append(math.ceil(prefix_tokens / block_size))
    return sorted(vals)


def modeled_residual_blocks(session_a_hash_ids, session_b_hash_ids, model_a, model_b,
                            trace_a, trace_b, shared_prefix_blocks, block_size, vocab_size):
    """Offline instrument check: synthesize two sessions, hash them the way vLLM will, and
    return the cross-session shared-prefix length in blocks. When the two are the same
    model this must equal shared_prefix_blocks (the modeled residual the agent measures on
    the box); when different models it must be 0 (no shared preamble)."""
    ta = synth_request_tokens(session_a_hash_ids, trace_a, model_a,
                              shared_prefix_blocks, block_size, vocab_size)
    tb = synth_request_tokens(session_b_hash_ids, trace_b, model_b,
                              shared_prefix_blocks, block_size, vocab_size)
    ha = vllm_chain_hashes(ta, block_size)
    hb = vllm_chain_hashes(tb, block_size)
    n = 0
    for x, y in zip(ha, hb, strict=False):
        if x != y:
            break
        n += 1
    return n
