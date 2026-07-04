# agentic-kv-bench
Policy benchmark for KV cache eviction/placement under agentic workloads.

Spec lives in docs/ — read before design decisions:
- docs/benchmark-spec.md   — the gap, scope (two lanes), adoption requirements, packaging
- docs/trace-schema.md     — trace format, lifecycle events, the five scenarios, tracegen
- docs/oracle.md           — Belady-style approximate oracle
- docs/policy-interface.md — policy interface, hint interface, baselines, the contender, metrics
- docs/validation.md       — production-engine (vLLM/KVBM) validation, upstream RFC path
- docs/EVAL_PLAN.md        — pre-registration skeleton

Depends on miniserve as one replay backend — dependency direction is bench -> engine only.
