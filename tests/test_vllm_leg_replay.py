"""Declared-prefix replay mode: the modeled cross-session sharing is exactly S blocks and
nothing leaks past it (docs/vllm-leg-design.md §3). This is the offline instrument check
for headline number one before any rental."""

from agentic_kv_bench.vllm_leg.replay import (
    RESIDUAL_LABEL,
    measured_shared_prefix_blocks,
    modeled_residual_blocks,
    synth_session,
)

BS, VOCAB, S = 16, 32_000, 5


def _sess(trace_id, tail):
    # a session whose blocks 1..S are the shared preamble and S+1.. are session-unique
    return list(range(1, S + 1)) + tail


def test_same_model_sessions_share_exactly_S_blocks():
    # two DIFFERENT traces, same model -> modeled residual == S (the shared preamble),
    # not more (no false sharing in the tails) and not less (preamble fully shared).
    a = _sess("trace_A", [90, 91, 92])
    b = _sess("trace_B", [90, 91, 92])  # identical LOCAL tail ids, but different trace
    n = modeled_residual_blocks(a, b, "opus", "opus", "trace_A", "trace_B", S, BS, VOCAB)
    assert n == S, f"expected shared prefix {S}, got {n}"


def test_different_models_share_nothing():
    # different model -> different SHARED namespace -> zero cross-session overlap.
    a = _sess("trace_A", [90])
    b = _sess("trace_B", [90])
    assert modeled_residual_blocks(a, b, "opus", "sonnet",
                                   "trace_A", "trace_B", S, BS, VOCAB) == 0


def test_local_tails_do_not_falsely_share():
    # same trace-local hash id in two DIFFERENT traces must NOT synthesize equal (that is
    # the fabrication trap). Beyond the shared prefix, overlap stops exactly at S.
    a = _sess("trace_A", [90, 91])
    b = _sess("trace_B", [90, 91])
    # blocks S+1 onward use ("LOCAL", trace_id, hash) so they differ despite equal ids.
    assert modeled_residual_blocks(a, b, "opus", "opus",
                                   "trace_A", "trace_B", S, BS, VOCAB) == S


def test_within_session_sharing_preserved_in_tail():
    # a repeated LOCAL block id within the SAME trace must synthesize identically (this is
    # what eviction waste / number two depends on).
    from agentic_kv_bench.vllm_leg.replay import synth_request_tokens
    t1 = synth_request_tokens([S + 1], "trace_A", "opus", S, BS, VOCAB)
    t2 = synth_request_tokens([S + 1], "trace_A", "opus", S, BS, VOCAB)
    assert t1 == t2
    # ...and differs across traces
    t3 = synth_request_tokens([S + 1], "trace_B", "opus", S, BS, VOCAB)
    assert t1 != t3


def test_synth_session_shapes_requests():
    trace = {
        "id": "trace_A", "models": ["opus"],
        "requests": [
            {"t": 0.0, "out": 10, "hash_ids": _sess("trace_A", [90])},
            {"t": 1.5, "out": 20, "hash_ids": _sess("trace_A", [90, 91])},
        ],
    }
    sess = synth_session(trace, S, BS, VOCAB)
    assert [r["t"] for r in sess] == [0.0, 1.5]
    assert [r["out"] for r in sess] == [10, 20]
    # request 2 is a strict prefix-extension of request 1 in token space (within-session
    # prefix reuse preserved for the band analysis)
    assert sess[1]["token_ids"][:len(sess[0]["token_ids"])] == sess[0]["token_ids"]


def test_measured_prefix_distribution_is_empirical():
    traces = [
        {"system_tokens": 2000, "tool_tokens": 12000},   # ceil(14000/64)=219
        {"system_tokens": 700, "tool_tokens": 8000},     # ceil(8700/64)=136
    ]
    vals = measured_shared_prefix_blocks(traces, block_size=64)
    assert vals == [136, 219]


def test_label_states_the_caveat():
    # the caveat is part of the number wherever it appears.
    assert "modeled" in RESIDUAL_LABEL and "does not preserve cross-session" in RESIDUAL_LABEL
