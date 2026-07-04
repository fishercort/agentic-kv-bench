# agentic-kv-bench — validation on a production engine (Phase 4)

The benchmark's findings come from a scratch engine (miniserve) plus a
simulator. This phase checks that they survive contact with a production
engine, and proposes the mechanism for them to land there permanently.

## The port

Port the winning policy to plug into vLLM via Dynamo's KV-event interface
(KVBM is pip-installable). Confirm the own-engine findings survive on a
production engine at real scale. Execution mode (c) in
`policy-interface.md` — the port reuses the same traces, so the comparison
is like-for-like.

## Headline charts

- p95/p99 TTFT and throughput vs baselines under memory pressure.
- Tokens-recomputed reduction.
- **High-value hit rate** — hits on expensive-to-recompute blocks, not just
  cheap ones. (This is the same metric reported per harness run; see
  `policy-interface.md`, Reported per run.)
- Percent-of-oracle.

Writeup discipline: mechanism first, numbers second.

## The upstream path: the RFC

An upstream contract proposal is how benchmark findings become production
behavior — a policy that only wins in a standalone harness changes nothing
about how engines actually manage KV. The proposal: an RFC to LMCache (first
choice — they are the community closest to this problem and already benchmark
agentic workloads) or vLLM, proposing the lifecycle-hint contract
(`policy-interface.md`, Lifecycle hint interface), with the head-to-head
comparison data as evidence.

Even an unmerged RFC puts the proposed contract and its supporting data in
front of the engine maintainers and makes the design public and referenceable;
a merged one makes the benchmark's findings part of a production engine's
behavior.
