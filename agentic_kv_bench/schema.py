"""The benchmark trace schema (docs/trace-schema.md), as dataclasses with JSONL
serialization. One JSONL line per request; requests sharing a session_id form a
conversation.
"""

import json
from dataclasses import asdict, dataclass, field

# The stable cross-version contract. Pinned canonical traces and any user's
# policy both target a schema_version; results are comparable only within one.
# Bump on any breaking change to the schema below.
SCHEMA_VERSION = 1

SpanKind = str  # system_prompt | history | tool_output | reasoning
Confidence = str  # measured | derived | residual


@dataclass(frozen=True)
class Span:
    span_id: str
    kind: SpanKind
    tokens: int
    confidence: Confidence  # how the kind was determined (Decision 2)
    shared_across_sessions: bool = False


@dataclass(frozen=True)
class LifecycleEvent:
    event: str  # span_close | compaction | subagent_terminate
    at_ms: int
    span_id: str | None = None
    scope: str | None = None
    blocks_dropped: int | None = None  # for compaction


@dataclass
class Request:
    req_id: str
    session_id: str
    arrival_ms: int
    spans: list[Span]
    output_tokens: int
    lifecycle_events: list[LifecycleEvent] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


@dataclass
class TraceStats:
    """Per-trace conversion report: the measured knobs and the confidence
    accounting a reviewer needs to trust the derivation."""

    schema_version: int
    session_id: str
    n_requests: int
    n_compactions: int
    # LITERAL-reuse ephemerality: fraction of blocks accessed in exactly one
    # prefix. For append-only agentic traces this measures PHYSICAL eviction
    # (near-zero until compaction), NOT the semantic "went useless" ephemerality
    # the benchmark motivates. Semantic ephemerality is a derived inference from
    # span kinds; see docs/trace-conversion.md open finding.
    physical_reuse_ephemeral_fraction: float
    reuse_count_histogram: dict[int, int]  # reuse_count -> block count
    kind_by_confidence: dict[str, dict[str, int]]  # kind -> confidence -> tokens

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))
