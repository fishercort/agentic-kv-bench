# agentic-kv-bench — policy interface, baselines, and the contender

## The falsifiable hypothesis

Under real memory pressure on real agentic traces, pricing the full
evict/offload/recompute decision with measured hardware costs beats pure
eviction scoring. The published policies all answer "which block do I drop?";
none question that dropping is the action. The contender policy prices the
four-way decision — evict, offload to a cheaper tier, recompute later, or refuse
admission — using the calibrated cost model. Either outcome is informative: a
positive result is a measured, reproducible improvement; a negative result is an
honest finding reported through the same benchmark.

## What is consolidation, what is new

Most of this benchmark is consolidation — reimplementing published policies
under one harness, against shared traces and a common oracle. The new pieces are
the measured migrate-vs-recompute cost model (no paper publishes these curves),
the lifecycle hint interface, and the economic joint policy.

## The policy interface

A policy is a class implementing:

- `on_access(block, meta, now)`
- `on_hint(event, now)`
- `evict(needed_bytes, now) -> [block_ids]`
- `place(block) -> tier`  (optional)

Anyone's policy can run against the benchmark, which is what makes this a shared
benchmark rather than a private eval. The interface is small enough to implement
in an afternoon and ships with a worked example.

## Lifecycle hint interface

The API contract: span open/close with lifecycle class
(durable | ephemeral | subagent | reasoning), pin/unpin, optional TTL. Wired
through the engine so hints attach to `BlockMeta`. The contract must degrade
gracefully to inference-only mode when hints are missing or wrong — that
graceful degradation is what makes it a real interface rather than a demo
assumption. Document the contract and its failure modes.

## Baselines

Reimplemented from the literature. Fidelity to the papers' exact scoring rules
matters — the authors may check, and a baseline is only fair if it matches the
source:

- **LRU** — the floor.
- **TTL-based eviction (Continuum, 2511.02230)** — keep KV alive through
  tool-call gaps via time-to-live; the simplest agentic-aware baseline.
- **GDSF-style cost-aware** — value = recompute_cost × reuse_prob / size.
- **WA-LRU (SAGA, 2605.00528)** — normalized recency + reuse probability + size.
- **Retired-cache lifecycle eviction (2605.06472)** — workflow-tagged blocks;
  termination messages trigger reclamation. This doubles as the hint-interface
  consumer — the termination message IS a lifecycle hint.

## The contender

The economic joint policy: at memory pressure, price all four actions — evict,
offload to a cheaper tier, plan recompute, refuse admission — using the measured
cost model, and take the cheapest. The published policies decide WHAT to
drop; this one decides WHETHER dropping is even the right action. Run it three
ways per the harness switches: with hints, hint-degraded, and inference-only.

## Execution modes

- (a) Simulator replay against the calibrated cost model, for fast sweeps.
- (b) Live replay through the miniserve engine, for end-to-end latency numbers.
- (c) Production-engine port target (vLLM / KVBM) reuses the same traces.

## Hint degradation switches

Per run: hints on / hints delayed by N ms / hints dropped with probability p /
hints off. This produces the robustness curve that makes the interface credible
as a real contract rather than a demo assumption.

## Reported per run

Percent-of-oracle, tokens recomputed, high-value hit rate (hits on
expensive-to-recompute blocks, not just cheap ones — also a production-validation
headline chart, see `validation.md`), p95/p99 TTFT (live mode), KV occupancy over time,
realized ephemeral fraction.

## Deliverable

The head-to-head the field lacks: all published policies plus the contender,
across memory pressure, on real and synthetic traces, all as percent-of-oracle.
Report wins and losses per the pre-registered plan (`EVAL_PLAN.md`).
