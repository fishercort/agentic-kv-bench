"""The replayer's acceptance test: token synthesis preserves prefix-sharing exactly
and creates no false sharing. This is load-bearing for
residual dedup: if synthesis broke sharing, vLLM's hashing would not reproduce the
cross-instance overlap the headline number is computed from.
"""

from agentic_kv_bench.vllm_leg.synth import (
    shared_prefix_len,
    synth_block_tokens,
    synth_request_tokens,
    vllm_chain_hashes,
)

BS, VOCAB = 4, 10_000


def _vllm_overlap(hash_chain_a, hash_chain_b):
    """Shared-prefix length as vLLM would see it: synthesize both, hash the tokens
    vLLM's way, compare the chained-hash prefixes."""
    ha = vllm_chain_hashes(synth_request_tokens(hash_chain_a, BS, VOCAB), BS)
    hb = vllm_chain_hashes(synth_request_tokens(hash_chain_b, BS, VOCAB), BS)
    return shared_prefix_len(ha, hb)


def test_synthesis_preserves_prefix_sharing():
    # sessions with KNOWN shared-prefix lengths (in trace content hashes)
    a = [10, 11, 12, 13]
    b = [10, 11, 99, 98]   # shares a 2-block prefix with a
    c = [50, 51, 52]       # shares nothing with a
    d = [10, 11, 12, 13, 77]  # a is a prefix of d
    cases = [(a, b, 2), (a, c, 0), (b, c, 0), (a, a, 4), (a, d, 4)]
    for x, y, expected in cases:
        assert shared_prefix_len(x, y) == expected, "trace-level sanity"
        # the claim: vLLM's chained-hash overlap == the trace's shared-prefix length
        assert _vllm_overlap(x, y) == expected, (
            f"synthesis broke sharing: trace prefix {expected}, "
            f"vLLM-hash prefix {_vllm_overlap(x, y)}"
        )


def test_no_false_sharing_distinct_blocks_distinct_tokens():
    # distinct content hashes must synthesize to distinct token blocks, or residual
    # dedup would be INFLATED by accidental over-sharing.
    blocks = {synth_block_tokens(h, 8, VOCAB) for h in range(500)}
    assert len(blocks) == 500  # no collisions


def test_divergent_block_ends_the_shared_prefix_exactly():
    # a and b agree for 3 blocks then diverge; the vLLM overlap must be exactly 3,
    # not 2 (under-share) or 4 (a divergent block hashing equal by accident).
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 9, 9]
    assert _vllm_overlap(a, b) == 3


def test_synthesis_is_deterministic_and_position_independent():
    # same content hash -> same tokens, regardless of where it sits in a chain.
    t1 = synth_block_tokens(42, BS, VOCAB)
    t2 = synth_block_tokens(42, BS, VOCAB)
    assert t1 == t2
    # the block for hash 42 is identical whether it is block 0 of one session or
    # block 2 of another (position-independence is what makes shared prefixes align).
    s1 = synth_request_tokens([42, 7], BS, VOCAB)[:BS]
    s2 = synth_request_tokens([9, 8, 42], BS, VOCAB)[2 * BS:3 * BS]
    assert tuple(s1) == tuple(s2) == t1
