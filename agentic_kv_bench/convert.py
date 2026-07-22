"""Convert the kv-cache-tester corpus into the benchmark schema.

Implements docs/trace-conversion.md. The derivation rules (block-hash reuse
into span kinds and lifecycle ground truth) are the contribution; the three
Decisions there govern ephemeral definition, confidence tiering, and subagent
deferral.

A source trace is a JSON object with a `requests` array. Normal requests carry
`hash_ids` (one content hash per BLOCK_TOKENS-token prompt block); subagent
requests (type "subagent") carry no hash_ids and a nested `requests` array.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

from agentic_kv_bench.schema import LifecycleEvent, Request, Span, TraceStats

BLOCK_TOKENS = 64  # kv-cache-tester block size


class UnexpectedTrace(Exception):
    """A trace violates a documented assumption; surfaced, never silently
    dropped (docs/trace-conversion.md, self-validation)."""


class SubagentTrace(Exception):
    """Trace contains subagent requests; deferred to the v2 pass (Decision 3).
    Detected and counted by the caller, not dropped."""


@dataclass
class _Block:
    kind: str
    confidence: str
    origin_turn: int
    appearances: list[int]  # request indices whose prefix contains this block


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
    """Kind + confidence for a block first appended at prefix position `pos`
    in a given request. Order matters: system region wins, then typed signals,
    then residual history (Decision 2 confidence tiers)."""
    if pos < sys_blocks:
        return "system_prompt", "measured"
    if produced_thinking_prev:
        # The prior turn emitted reasoning; its output blocks land in this
        # prefix. Derived, and low confidence: reasoning is not cleanly
        # separable from the response without more signal.
        return "reasoning", "derived"
    if has_tool_result:
        return "tool_output", "derived"
    return "history", "residual"


def convert_trace(trace: dict) -> tuple[list[Request], TraceStats]:
    """One source trace -> schema requests + a per-trace stats report.

    Raises SubagentTrace for subagent-bearing traces (Decision 3) and
    UnexpectedTrace for any documented-assumption violation.
    """
    reqs = _normal_requests(trace)
    session_id = trace["id"]
    sys_blocks = -(-trace.get("system_tokens", 0) // BLOCK_TOKENS)
    n = len(reqs)

    blocks: dict[int, _Block] = {}  # block hash -> info
    prev_prefix: list[int] = []
    n_compactions = 0
    schema_requests: list[Request] = []

    for i, r in enumerate(reqs):
        prefix = r["hash_ids"]
        arrival_ms = int(round(r["t"] * 1000))
        events: list[LifecycleEvent] = []

        # Compaction: prefix diverges before its trailing (partial) block.
        # Compare up to the shorter length minus the boundary block.
        compact_at = None
        m = min(len(prev_prefix), len(prefix))
        if m > 1 and prev_prefix[: m - 1] != prefix[: m - 1]:
            compact_at = next(
                j for j in range(m - 1) if prev_prefix[j] != prefix[j]
            )
        dropped = 0
        if compact_at is not None and compact_at < len(prev_prefix) - 1:
            dropped = len(prev_prefix) - compact_at
            n_compactions += 1
            events.append(
                LifecycleEvent(
                    event="compaction", at_ms=arrival_ms, blocks_dropped=dropped
                )
            )

        has_tool_result = "tool_result" in r.get("input_types", [])
        produced_thinking_prev = i > 0 and (
            "thinking" in reqs[i - 1].get("output_types", [])
        )

        # Register/appearance-track every block in this prefix. New blocks
        # (not seen before) get a kind at first append.
        for pos, h in enumerate(prefix):
            b = blocks.get(h)
            if b is None:
                kind, conf = _classify_new_block(
                    pos, sys_blocks, has_tool_result, produced_thinking_prev
                )
                b = _Block(kind=kind, confidence=conf, origin_turn=i, appearances=[])
                blocks[h] = b
            b.appearances.append(i)

        # Emit this request's spans: maximal runs of same (kind, confidence).
        spans = _runs_to_spans(prefix, blocks, session_id, i)
        schema_requests.append(
            Request(
                req_id=f"{session_id}-r{i:04d}",
                session_id=session_id,
                arrival_ms=arrival_ms,
                spans=spans,
                output_tokens=r.get("out", 0),
                lifecycle_events=events,
            )
        )
        prev_prefix = prefix

    stats = _finalize(session_id, n, n_compactions, blocks)
    return schema_requests, stats


def _runs_to_spans(
    prefix: list[int], blocks: dict[int, _Block], session_id: str, turn: int
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
                    tokens=(j - run_start) * BLOCK_TOKENS,
                    confidence=b.confidence,
                )
            )
            run_start = j
    return spans


def _finalize(session_id, n, n_compactions, blocks) -> TraceStats:
    # Reuse count = number of DISTINCT requests whose prefix contains the block.
    reuse_hist: Counter[int] = Counter()
    ephem_blocks = 0
    denom_blocks = 0  # blocks with a reuse opportunity (not final-turn-only)
    # kind -> confidence -> tokens, counting each block ONCE by its kind (a
    # corpus-composition measure, not per-request-appearance which would
    # inflate durable kinds by their turn count).
    kbc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for b in blocks.values():
        distinct = len(set(b.appearances))
        reuse_hist[distinct] += 1
        kbc[b.kind][b.confidence] += BLOCK_TOKENS
        last = max(b.appearances)
        if last == n - 1 and distinct == 1:
            continue  # final-request-only: undetermined, excluded (Decision 1)
        denom_blocks += 1
        if distinct == 1:
            ephem_blocks += 1
    ephemeral_fraction = (ephem_blocks / denom_blocks) if denom_blocks else 0.0

    return TraceStats(
        session_id=session_id,
        n_requests=n,
        n_compactions=n_compactions,
        physical_reuse_ephemeral_fraction=ephemeral_fraction,
        reuse_count_histogram=dict(sorted(reuse_hist.items())),
        kind_by_confidence={k: dict(v) for k, v in kbc.items()},
    )
