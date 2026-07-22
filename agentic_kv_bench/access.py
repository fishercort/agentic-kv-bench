"""Adapter: source trace -> the harness's execution form (RequestAccess).

The schema (Request/Span) is the portable, human- and policy-facing artifact;
the AccessTrace is the simulator's execution form. Both derive from the same
`walk_source`, so block kinds and lifecycle events are identical across them.

Block ids are the source content hashes (already anonymized in the corpus) and
are stable across requests, which is exactly what the cache simulation needs to
model reuse. The trailing partial block is priced at the full block size (a
minor overcount noted in docs/trace-conversion.md).
"""

from dataclasses import asdict

from agentic_kv_bench.convert import walk_source
from agentic_kv_bench.harness import BlockRef, RequestAccess


def access_from_source(trace: dict) -> list[RequestAccess]:
    """Source trace -> replayable accesses. Raises SubagentTrace / UnexpectedTrace
    (a caller replaying a corpus catches SubagentTrace to defer, per Decision 3).

    Block ids are namespaced by session (`session#hash`) so that interleaving
    multiple sessions in one cache never collides two sessions' blocks, even if
    the source hashes happen to coincide (they are per-conversation, local
    scope)."""
    sw = walk_source(trace)
    accesses: list[RequestAccess] = []
    for w in sw.walks:
        blocks = [
            BlockRef(
                block_id=f"{sw.session_id}#{h}",
                kind=sw.blocks[h].kind,
                size_tokens=sw.block_tokens,
            )
            for h in w.prefix_ids
        ]
        accesses.append(
            RequestAccess(
                arrival_ms=w.arrival_ms,
                blocks=blocks,
                lifecycle_events=[asdict(e) for e in w.events],
            )
        )
    return accesses
