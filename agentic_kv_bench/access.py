"""Adapter: source trace -> the harness's execution form (RequestAccess).

The schema (Request/Span) is the portable, human- and policy-facing artifact;
the AccessTrace is the simulator's execution form. Both derive from the same
`walk_source`, so block kinds are identical across them. Lifecycle events differ
in ONE deliberate way: the schema keeps compaction as a portable count, while
the execution form RESOLVES each compaction into a `retire` hint naming the
concrete cache block_ids it killed (a hint is only actionable if it names blocks
in the id space the policy sees, and that id space, coarsened and session-
namespaced, is minted here, not in the schema). See docs/hint-interface.md.

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

    # First pass: mint each request's cache block_ids (coarsened, namespaced).
    per_request_blocks: list[list[BlockRef]] = []
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
        per_request_blocks.append(blocks)

    # Ground-truth last use per block_id: the retirement hint must be TRUE (a
    # block it names must never be accessed again), so a policy that trusts the
    # hint is never punished for a correct signal. Degradation is injected by the
    # harness switches (delay/drop), not by lying about which blocks are dead.
    last_use: dict[object, int] = {}
    for i, blocks in enumerate(per_request_blocks):
        for b in blocks:
            last_use[b.block_id] = i

    # Second pass: at each compaction, emit a retire hint naming the blocks the
    # previous prefix held that are now dead (last used before this request).
    # Retirement is gated on the real compaction event, not on arbitrary death,
    # so the hint models a signal a serving framework actually has (it performed
    # the compaction) rather than an oracle of every block's future.
    accesses: list[RequestAccess] = []
    prev_blocks: list[BlockRef] = []
    for i, (w, blocks) in enumerate(zip(sw.walks, per_request_blocks, strict=True)):
        lifecycle: list[dict] = []
        if any(e.event == "compaction" for e in w.events):
            retired = [b.block_id for b in prev_blocks if last_use[b.block_id] < i]
            if retired:
                lifecycle.append(
                    {"event": "retire", "at_ms": w.arrival_ms, "block_ids": retired}
                )
        accesses.append(
            RequestAccess(arrival_ms=w.arrival_ms, blocks=blocks, lifecycle_events=lifecycle)
        )
        prev_blocks = blocks
    return accesses
