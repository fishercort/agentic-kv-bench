# agentic-kv-bench — trace schema, scenarios, and generator

## What a trace is

A trace is a JSONL file: a time-ordered stream of requests grouped into
sessions, with token-level span structure and lifecycle ground truth.

```jsonc
// one line per request
{
  "req_id": "r-0421",
  "session_id": "s-017",          // requests in a session share prefix history
  "arrival_ms": 184223,            // absolute, from trace start
  "spans": [
    // ordered token spans composing the prompt; KV blocks inherit span identity
    {"span_id": "sys-A",   "kind": "system_prompt", "tokens": 1200, "shared_across_sessions": true},
    {"span_id": "hist-17", "kind": "history",       "tokens": 3400},
    {"span_id": "tool-88", "kind": "tool_output",   "tokens": 2100},
    {"span_id": "cot-31",  "kind": "reasoning",     "tokens": 1800}
  ],
  "output_tokens": 350,
  "lifecycle_events": [
    // ground truth emitted by the generator; the oracle sees all of it,
    // hint-mode policies see it only when the scenario grants hints
    {"event": "span_close",        "span_id": "cot-31",  "at_ms": 185002},
    {"event": "subagent_terminate","scope": "sub-9",     "at_ms": 189500}
  ]
}
```

Ground truth lives in the trace, not in the policy. This is what makes three
things possible with one artifact: the Belady-style oracle (reads all future
events), hint-mode policies (read events as they occur), and inference-only
policies (read nothing, must guess from access patterns).

Definition of the headline knob, stated precisely so it is measurable:
**ephemeral fraction = share of KV bytes whose span is never re-accessed after
its close event.** The generator targets a value; the harness reports the
realized value per trace. Sweep axis: 0.1 → 0.7.

## The five scenarios

Each anchored to a workload pattern a practitioner will recognize on sight.
Parameters are calibrated where public data exists, and every parameter is
published in the trace header.

1. **Multi-turn tool agent** (SWE-agent / coding-agent shape). Long-lived
   session; each turn appends tool output (large, mostly cold after next turn)
   and reasoning spans (dead after the turn). Calibrate turn counts and
   tool-output sizes against public SWE-bench trajectory data.
2. **Subagent fan-out** (orchestrator–worker). An orchestrator session spawns
   3–12 subagents sharing the orchestrator prefix; subagents terminate and their
   entire KV goes dead at a known instant. The sharpest test of the hint
   interface: `subagent_terminate` should trigger immediate reclamation.
3. **Reasoning loops.** Requests with think-spans of 30–50% of generated tokens
   that close before the final answer (the pattern NVIDIA called out: ~40% of
   tokens becoming useless when the loop closes). Tests reasoning-span eviction
   priority.
4. **Multi-tenant support bots.** Many short sessions, a handful of heavy shared
   system prompts, Zipf-distributed tenant popularity. Mostly-durable workload;
   tests that lifecycle awareness does not regress the classic prefix-sharing
   case that RadixAttention already serves well.
5. **Uniform chat (control).** ShareGPT-shaped multi-turn chat, near-zero
   ephemeral content, no subagents. **Designed so the lifecycle policy should NOT
   meaningfully beat cost-aware or even LRU.** Reporting this honestly is the
   single strongest anti-rigging move in the benchmark.

## Sourcing

Real traces are the anchor, synthesis is the supplement. The kv-cache-tester
corpus (739 anonymized Claude Code conversation traces, used by LMCache's May
2026 MI300X benchmark) is the primary workload — convert it into the schema
above, crediting the source. The synthetic generator remains for what real
traces cannot provide: the controllable ephemeral-fraction sweep, scenario
isolation (fan-out-only, reasoning-only), and the Scenario 5 control. Secondary
calibration sources for the generator: Mooncake's published production traces and
Azure's public LLM inference traces for arrival processes; SWE-bench trajectories
and ShareGPT for turn/session distributions. Where a parameter has no public
anchor, it is declared as an assumption in the trace header rather than buried.

Note on lifecycle ground truth for real traces: Claude Code conversations carry
reconstructable structure (subagent spawns, tool calls, turn boundaries).
Deriving span annotations from that structure is part of the conversion work —
document the derivation rules; they are themselves a contribution.

## The generator

`tracegen` — a deterministic, seeded Python package.

- Input: scenario name + parameter overrides + seed. Output: trace JSONL + a
  header block recording every parameter, the seed, the calibration source per
  parameter, and the realized ephemeral fraction.
- Arrival process: Poisson baseline with burst option (Markov-modulated), rates
  calibrated to the public Azure traces.
- Determinism requirement: same seed → byte-identical trace. Reviewers can
  regenerate everything.
- Ships with 15–25 canonical pre-generated traces (5 scenarios × ephemeral sweep
  points) so results are reproducible without running the generator.
