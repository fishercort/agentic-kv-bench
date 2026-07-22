"""Convert the kv-cache-tester corpus into the benchmark schema.

Implements docs/trace-conversion.md. The derivation rules (block-hash reuse
into span kinds and lifecycle ground truth) are the contribution; the three
Decisions there govern ephemeral definition, confidence tiering, and subagent
deferral.

A source trace is a JSON object with a `requests` array. Normal requests carry
`hash_ids` (one content hash per block_size-token prompt block); subagent
requests (type "subagent") carry no hash_ids and a nested `requests` array.

`walk_source` is the shared block-walk both the schema builder (convert_trace)
and the harness adapter (access.access_from_source) consume, so the block-kind
derivation lives in exactly one place.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from agentic_kv_bench.schema import (
    SCHEMA_VERSION,
    LifecycleEvent,
    Request,
    Span,
    TraceStats,
)

# Fallback only. The block size is READ from each source trace's block_size
# field (portability: the data states its own geometry, we do not assume it).
DEFAULT_BLOCK_TOKENS = 64


class UnexpectedTrace(Exception):
    """A trace violates a documented assumption; surfaced, never silently
    dropped (docs/trace-conversion.md, self-validation)."""


class SubagentTrace(Exception):
    """Trace contains subagent requests; deferred to the v2 pass (Decision 3).
    Detected and counted by the caller, not dropped."""


@dataclass
class Block:
    """A distinct block-hash and the derivation attached to it at first sight."""

    kind: str
    confidence: str
    origin_turn: int
    appearances: list[int]  # request indices whose prefix contains this block


@dataclass
class ReqWalk:
    """One request's replay-relevant facts, produced by walk_source."""

    arrival_ms: int
    prefix_ids: list[int]  # the block hashes this request's prefix accesses
    events: list[LifecycleEvent] = field(default_factory=list)


@dataclass
class SourceWalk:
    session_id: str
    block_tokens: int
    n_requests: int
    n_compactions: int
    blocks: dict[int, Block]  # block hash -> derivation
    walks: list[ReqWalk]


def _normal_requests(trace: dict) -> list[dict]:
    reqs = trace["requests"]
    if any(r.get("type") == "subagent" for r in reqs):
        raise SubagentTrace(f"{trace['id']} has subagent requests")
    for r in reqs:
        if "hash_ids" not in r:
            raise UnexpectedTrace(f"{trace['id']} request without hash_ids or subagent type")
    return reqs


def _classify_new_block(
    pos: int, sys_blocks: int, has_tool_result: bool, produced_thinking_prev: bool
) -> tuple[str, str]:
    """Kind + confidence for a block first appended at prefix position `pos`.
    Order matters: system region wins, then typed signals, then residual
    history (Decision 2 confidence tiers)."""
    if pos < sys_blocks:
        return "system_prompt", "measured"
    if produced_thinking_prev:
        # The prior turn emitted reasoning; its output blocks land in this
        # prefix. Derived, low confidence: reasoning is not cleanly separable
        # from the response without more signal (a known open finding).
        return "reasoning", "derived"
    if has_tool_result:
        return "tool_output", "derived"
    return "history", "residual"


def walk_source(trace: dict) -> SourceWalk:
    """The shared block-walk: classify blocks, detect compaction, record each
    request's accessed prefix. Raises SubagentTrace / UnexpectedTrace."""
    reqs = _normal_requests(trace)
    session_id = trace["id"]
    block_tokens = trace.get("block_size", DEFAULT_BLOCK_TOKENS)
    sys_blocks = -(-trace.get("system_tokens", 0) // block_tokens)

    blocks: dict[int, Block] = {}
    walks: list[ReqWalk] = []
    prev_prefix: list[int] = []
    n_compactions = 0

    for i, r in enumerate(reqs):
        prefix = r["hash_ids"]
        arrival_ms = int(round(r["t"] * 1000))
        events: list[LifecycleEvent] = []

        # Compaction: prefix diverges before its trailing (partial) block.
        m = min(len(prev_prefix), len(prefix))
        if m > 1 and prev_prefix[: m - 1] != prefix[: m - 1]:
            compact_at = next(j for j in range(m - 1) if prev_prefix[j] != prefix[j])
            if compact_at < len(prev_prefix) - 1:
                n_compactions += 1
                events.append(
                    LifecycleEvent(
                        event="compaction",
                        at_ms=arrival_ms,
                        blocks_dropped=len(prev_prefix) - compact_at,
                    )
                )

        has_tool_result = "tool_result" in r.get("input_types", [])
        produced_thinking_prev = i > 0 and (
            "thinking" in reqs[i - 1].get("output_types", [])
        )
        for pos, h in enumerate(prefix):
            b = blocks.get(h)
            if b is None:
                kind, conf = _classify_new_block(
                    pos, sys_blocks, has_tool_result, produced_thinking_prev
                )
                b = Block(kind=kind, confidence=conf, origin_turn=i, appearances=[])
                blocks[h] = b
            b.appearances.append(i)

        walks.append(ReqWalk(arrival_ms=arrival_ms, prefix_ids=prefix, events=events))
        prev_prefix = prefix

    return SourceWalk(
        session_id=session_id,
        block_tokens=block_tokens,
        n_requests=len(reqs),
        n_compactions=n_compactions,
        blocks=blocks,
        walks=walks,
    )


def convert_trace(trace: dict) -> tuple[list[Request], TraceStats]:
    """One source trace -> schema requests + a per-trace stats report."""
    sw = walk_source(trace)
    reqs = trace["requests"]
    schema_requests = [
        Request(
            req_id=f"{sw.session_id}-r{i:04d}",
            session_id=sw.session_id,
            arrival_ms=w.arrival_ms,
            spans=_runs_to_spans(w.prefix_ids, sw.blocks, sw.session_id, i, sw.block_tokens),
            output_tokens=reqs[i].get("out", 0),
            lifecycle_events=w.events,
        )
        for i, w in enumerate(sw.walks)
    ]
    return schema_requests, _finalize(sw)


def _runs_to_spans(
    prefix: list[int], blocks: dict[int, Block], session_id: str, turn: int,
    block_tokens: int,
) -> list[Span]:
    spans: list[Span] = []
    run_start = 0
    for j in range(1, len(prefix) + 1):
        end = j == len(prefix)
        same = (not end) and (
            blocks[prefix[j]].kind == blocks[prefix[run_start]].kind
            and blocks[prefix[j]].confidence == blocks[prefix[run_start]].confidence
        )
        if not same:
            b = blocks[prefix[run_start]]
            spans.append(
                Span(
                    span_id=f"{session_id}-t{turn}-s{len(spans)}",
                    kind=b.kind,
                    tokens=(j - run_start) * block_tokens,
                    confidence=b.confidence,
                )
            )
            run_start = j
    return spans


def _finalize(sw: SourceWalk) -> TraceStats:
    # Reuse count = number of DISTINCT requests whose prefix contains the block.
    reuse_hist: Counter[int] = Counter()
    ephem_blocks = 0
    denom_blocks = 0  # blocks with a reuse opportunity (not final-turn-only)
    # kind -> confidence -> tokens, counting each block ONCE (corpus composition,
    # not per-request-appearance which inflates durable kinds by their turn count).
    kbc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for b in sw.blocks.values():
        distinct = len(set(b.appearances))
        reuse_hist[distinct] += 1
        kbc[b.kind][b.confidence] += sw.block_tokens
        if max(b.appearances) == sw.n_requests - 1 and distinct == 1:
            continue  # final-request-only: undetermined, excluded (Decision 1)
        denom_blocks += 1
        if distinct == 1:
            ephem_blocks += 1

    return TraceStats(
        schema_version=SCHEMA_VERSION,
        session_id=sw.session_id,
        n_requests=sw.n_requests,
        n_compactions=sw.n_compactions,
        physical_reuse_ephemeral_fraction=(
            ephem_blocks / denom_blocks if denom_blocks else 0.0
        ),
        reuse_count_histogram=dict(sorted(reuse_hist.items())),
        kind_by_confidence={k: dict(v) for k, v in kbc.items()},
    )
