# Trace conversion: kv-cache-tester into the benchmark schema

Converts the public kv-cache-tester corpus (739 anonymized Claude Code
conversation traces, 59k requests, Apache-2.0) into the benchmark trace schema
(trace-schema.md). The derivation rules below, block-hash reuse into span kinds
and lifecycle ground truth, are themselves a contribution and are documented as
such.

Design before converter code, per the Phase 2 lesson. The source-format claims
here were verified against a 12-trace sample spanning the corpus (2026-07-21),
not a single file, after the single-file version of this design proved wrong on
three load-bearing points.

## Source format (verified, 12-trace sample)

Each source file is one conversation trace, a JSON object: `id`, `models`,
`block_size` (64), `hash_id_scope`, `tool_tokens`, `system_tokens`, `totals`
(parent/subagent token counts, `subagent_count`), and `requests`.

A request is one of two shapes, distinguished by `type`:

- **Normal request** (`type` in {n, s}, both processed identically): `t`
  (arrival seconds from trace start), `in`, `out`, `hash_ids` (one content hash
  per 64-token prompt block), `input_types` (subset of {text, tool_result}),
  `output_types` (subset of {text, tool_use, thinking}), `stop` (subset of
  {tool_use, end_turn, empty}). Optional and variable: `api_time`, `ttft`,
  `think_time`. The processing rule keys on presence of `hash_ids`, not on the
  type letter, because the letter set is not closed.
- **Subagent request** (`type` = subagent): no `hash_ids`; carries `agent_id`,
  `subagent_type`, `duration_ms`, `status` (completed), `total_tokens`,
  `tool_use_count`, and a nested `requests` array of normal requests (with
  their own hash_ids). Nesting is one level in the sample (no sub-subagents).

## Verified corpus properties (and the ones that broke the naive design)

- **Growth is append-MOSTLY, not append-only.** Two distinct exceptions:
  - Boundary rehash: the trailing partial block re-hashes every turn as it
    fills. Cosmetic; the delta rule ignores the final block.
  - Context compaction: a prefix that diverges well before its tail is a real
    compaction event (history dropped, a shared root kept). Sparse (0 to 4 per
    trace in the sample) but structurally real, and modeled as a bulk
    span_close rather than rejected.
- **Scope is local across the sample.** Block hashes are per-conversation, so
  cross-session sharing is NOT measurable by comparing hashes across traces.
  Consequence below (Scenario 4).
- **Field presence varies.** api_time, ttft, and a non-empty stop are all
  optional. The parser tolerates absence.
- **Size range is wide.** 8 to 170+ requests per trace, 100 KB to 4 MB per
  file. The converter streams and must not assume a small trace.

## Target

The schema in trace-schema.md: requests grouped into sessions, each request a
list of token spans (span_id, kind in {system_prompt, history, tool_output,
reasoning}, tokens, shared_across_sessions) plus output_tokens and
lifecycle_events (span_close, subagent_terminate). One source trace maps to one
session; each source normal request maps to one schema request; each subagent
maps to a nested request group scoped by agent_id.

## The derivation rules (the contribution)

Stated with an explicit measured-vs-inferred split, because over-claiming
inferred structure is the failure mode this benchmark exists to avoid.

### Directly measured (high confidence)

- **arrival_ms** = `t` * 1000; **output_tokens** = `out`.
- **Prefix structure** = `hash_ids`, at 64-token block granularity, with the
  trailing partial block excluded from reuse comparison.
- **Ephemeral fraction** (headline knob), measured not targeted: a block is
  ephemeral if, after its close, it never reappears in a later prefix.
  Compaction makes close a first-class event (see below). Realized value
  reported per trace.
- **Compaction events**: a prefix divergence before the trailing block. Emits a
  bulk span_close for the dropped blocks at that request. This is a measured
  agentic lifecycle event, not noise.

### Derived from typed signals (medium confidence, rules documented)

- **system_prompt span**: leading `system_tokens` worth of blocks of the first
  request; stable across the session.
- **tool_output span**: when a request has `tool_result` in `input_types`, the
  blocks appended since the previous prefix (the hash_ids delta, boundary block
  excluded) are the tool output.
- **reasoning span**: when a request has `thinking` in `output_types`, that turn
  generated reasoning; persistence is then measured from whether those blocks
  appear in the next prefix (the ephemeral-reasoning pattern).
- **history span**: the residual kind, persistent prior-turn content not
  attributable to system, tool, or reasoning.

### Lifecycle events

- **span_close**: at the last request whose prefix contains a span's blocks, or
  at a compaction event that drops them.
- **subagent_terminate**: at the subagent request's position, using
  `status` = completed and `duration_ms` for timing. Scoped by agent_id. This
  is the sharpest hint-interface test and gets its own derivation care.

## Scenario mapping and the one honest limitation

The corpus is Scenario-1 material by nature (multi-turn tool agents; trace_0001
is 34/41 tool_use turns). Reasoning loops (Scenario 3) come from traces with
high `thinking` density. Subagent fan-out (Scenario 2) comes from the 19
subagent-bearing traces. **Scenario 4 (multi-tenant shared system prompts)
cannot be built from this corpus**, because local hash scope makes cross-session
sharing unmeasurable; Scenario 4 stays synthetic, or uses system-prompt content
identity if the corpus ever ships a global-scope variant. Stated as a limitation
rather than papered over.

## Open finding (surfaced by running the converter on real traces)

The "ephemeral fraction is directly measured" claim above is wrong for this
corpus, and running the v1 converter on 12 real traces proved it. Because
agentic prefixes are append-only, semantically dead KV (a reasoning block that
is useless after its turn) stays PHYSICALLY in the prompt prefix of every later
turn until a compaction event drops it. So literal reuse-count reads near-zero
ephemeral (0 to 9 percent in the sample), while the span-kind table shows the
real signal is huge (reasoning is 78 to 91 percent of tokens in reasoning-heavy
traces). Two consequences, both open decisions:

1. **Ephemerality is two distinct metrics, not one.** Physical-eviction
   ephemerality (literal reuse; near-zero until compaction) is measured and is
   now named `physical_reuse_ephemeral_fraction`. Semantic ephemerality (KV
   that went useless, the metric the benchmark motivates) is a DERIVED
   inference from span kinds plus compaction, not a measurement. The headline
   knob must be the semantic one, reported at derived confidence, with the
   physical one alongside. This corrects the "directly measured" claim above.
2. **Reasoning derivation is greedy.** Tagging all new content after a thinking
   turn as reasoning sweeps in the response and next tool call, inflating the
   reasoning share. A tighter rule is needed (for example, bound the reasoning
   span by the produced-thinking token count when available).

The v1 converter is committed and tested; these two are the next decisions
before the ephemeral metric graduates.

## What consumes the output

The simulator replays these traces against the cost model. Per the Phase 2
verdict, the v1 crossover is degenerate (launch-bound), so the simulator sweeps
recompute cost as a parameter and substitutes the calibrated crossover when
Phase 4 supplies it.

## Converter self-validation (the bulletproofing requirement)

The converter runs a validation pass over all 739 traces and fails loud on any
assumption this design makes that the data violates, the check_invariants
pattern applied to data:

- every request is a recognized shape (has hash_ids, or is a subagent);
- growth is append-mostly (every non-compaction divergence is at the trailing
  block only);
- compaction counts, subagent counts, and realized ephemeral fractions are
  reported per trace and in aggregate;
- token accounting reconciles (`in` within one block of hash_ids * 64).

An unrecognized type, a mid-prefix divergence that is neither boundary nor clean
compaction, or a global-scope trace is surfaced, not silently dropped. Silent
drops are how a benchmark quietly stops representing its corpus.

## Decisions (2026-07-21)

1. **Ephemeral definition (the headline metric).** A block is ephemeral if its
   hash appears in exactly one request's prefix (reuse-count 1) AND that
   appearance is before the final request (it had a reuse opportunity).
   Re-warming resolves for free: a dropped-then-returned block has count >= 2,
   so it is not ephemeral. Final-request-only blocks are undetermined (no reuse
   opportunity) and excluded from both numerator and denominator. Conservative
   by construction (undercounts). The full reuse-count distribution is also
   emitted, so the single fraction is not the only view.
2. **Confidence tiering.** Every span carries confidence in {measured, derived,
   residual}; the converter emits an aggregate kind-by-confidence table.
   system_prompt is measured (from system_tokens); tool_output and reasoning are
   derived (from typed signals); history is residual.
3. **Subagents deferred, never dropped.** v1 converts the non-subagent traces;
   subagent-bearing traces are detected, counted, and explicitly skipped with a
   logged reason, deferred to a v2 pass with the subagent_terminate derivation.
   Self-validation asserts the skip count matches the expected count.
