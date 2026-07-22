"""Adapter: source trace -> the harness's execution form (RequestAccess).

The schema (Request/Span) is the portable, human- and policy-facing artifact;
the AccessTrace is the simulator's execution form. Both derive from the same
`walk_source`, so block kinds and lifecycle events are identical across them.

Block granularity is a SIMULATION parameter, not an engine constraint. The
source hashes 64-token blocks; `sim_block_tokens` regroups them into coarser
simulated blocks (e.g. 256), which cuts resident-block counts proportionally
and is the first-order performance lever (EVAL_PLAN.md, sweep grid). Policy
RANKINGS are granularity-invariant because every policy faces the same
quantization; the eval pins one sensitivity check to confirm.

Block ids are (session_id, tuple-of-constituent-source-hashes): namespaced per
session so interleaved sessions never collide, and content-identity so the same
region in two prefixes is the same block (reuse works) and ids are deterministic
across runs (pinned traces reproduce).
"""

from dataclasses import asdict

from agentic_kv_bench.convert import walk_source
from agentic_kv_bench.harness import BlockRef, RequestAccess


def access_from_source(
    trace: dict, sim_block_tokens: int | None = None
) -> list[RequestAccess]:
    """Source trace -> replayable accesses. Raises SubagentTrace / UnexpectedTrace
    (a caller replaying a corpus catches SubagentTrace to defer, per Decision 3).

    sim_block_tokens coarsens the simulated block size; None or the source size
    means no coarsening. Must be a multiple of the source block size.
    """
    sw = walk_source(trace)
    src_tokens = sw.block_tokens
    if sim_block_tokens is None:
        sim_block_tokens = src_tokens
    if sim_block_tokens % src_tokens != 0:
        raise ValueError(
            f"sim_block_tokens ({sim_block_tokens}) must be a multiple of the "
            f"source block size ({src_tokens})"
        )
    group = sim_block_tokens // src_tokens

    accesses: list[RequestAccess] = []
    for w in sw.walks:
        prefix = w.prefix_ids
        blocks = []
        for start in range(0, len(prefix), group):
            grp = tuple(prefix[start : start + group])
            blocks.append(
                BlockRef(
                    block_id=(sw.session_id, grp),
                    kind=sw.blocks[grp[0]].kind,  # the group's leading kind
                    size_tokens=len(grp) * src_tokens,
                )
            )
        accesses.append(
            RequestAccess(
                arrival_ms=w.arrival_ms,
                blocks=blocks,
                lifecycle_events=[asdict(e) for e in w.events],
            )
        )
    return accesses
