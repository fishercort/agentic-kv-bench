"""Deterministic token synthesis for the vLLM-leg replayer.

The corpus traces are metadata (per-block content hashes, lengths, structure), not
text. To replay through real vLLM we synthesize token sequences whose PREFIX-SHARING
matches the trace exactly, so vLLM's real prefix-chained hashing reproduces the
cross-instance overlap that residual dedup (headline number two) is computed from.
The report phrase is "real session structure,
synthesized content", not "real traffic".

The invariant (this module's whole job): `T(content_hash) -> block_size token ids`
is deterministic, position-independent, and collision-resistant. So shared trace-hash
prefixes produce identical token prefixes -> identical vLLM chained hashes; a divergent
block produces different tokens -> a different chained hash. Preserve sharing exactly,
create no false sharing. `vllm_chain_hashes` reproduces vLLM's chaining so the invariant
can be tested without a GPU.
"""

import hashlib
import random


def synth_block_tokens(content_hash: object, block_size: int, vocab_size: int) -> tuple:
    """Deterministic, collision-resistant token block for a trace content hash.
    Depends ONLY on the content hash (not position), so equal-content blocks
    anywhere synthesize identically."""
    seed = hashlib.sha256(repr(content_hash).encode()).digest()
    rng = random.Random(seed)
    return tuple(rng.randrange(vocab_size) for _ in range(block_size))


def synth_request_tokens(hash_chain, block_size: int, vocab_size: int) -> list:
    """Token ids for a request = concatenation of T over its block-hash chain."""
    toks: list = []
    for h in hash_chain:
        toks.extend(synth_block_tokens(h, block_size, vocab_size))
    return toks


def vllm_chain_hashes(token_ids, block_size: int) -> list:
    """Reproduce vLLM's prefix-chained block hashing (kv_cache_utils.hash_block_tokens):
    each full block's hash = H(parent_hash, block_tokens). Partial trailing block is not
    hashed (vLLM only prefix-caches full blocks). Used to test the synthesis invariant
    without a GPU; the real run uses vLLM's own hashes off the event stream."""
    hashes = []
    parent = b"\x00" * 32  # NONE_HASH stand-in
    n_full = len(token_ids) // block_size
    for b in range(n_full):
        block = tuple(token_ids[b * block_size:(b + 1) * block_size])
        parent = hashlib.sha256(parent + repr(block).encode()).digest()
        hashes.append(parent)
    return hashes


def shared_prefix_len(a, b) -> int:
    """Length of the identical leading run of two sequences (block-granular overlap)."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n
