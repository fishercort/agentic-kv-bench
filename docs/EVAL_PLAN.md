# EVAL_PLAN — pre-registration

Committed before running the full sweep. It fixes the scenarios, the sweep grid,
the metrics, and the predictions in advance, so reported results cannot be
retrofitted to flatter any policy. Everything in the grid is reported afterward,
wins and losses. A benchmark whose author predicted and published where their own
method loses is far harder to dismiss as rigged.

_This is a skeleton. Fill every TBD before the run, then freeze the file._

## Scenarios (fixed in advance)

1. Multi-turn tool agent
2. Subagent fan-out
3. Reasoning loops
4. Multi-tenant support bots
5. Uniform chat (control)

See `trace-schema.md` for the definitions.

## Sweep grid

- Ephemeral fraction: 0.1 → 0.7  (grid points: TBD)
- Memory-pressure levels: TBD
- Hint modes: on / delayed by N ms / dropped w.p. p / off
- Policies: LRU, Continuum TTL, GDSF-style cost-aware, WA-LRU, retired-cache
  lifecycle, economic joint
- Seeds: TBD

## Metrics

Percent-of-oracle (headline), tokens recomputed, high-value hit rate, p95/p99
TTFT (live mode), KV occupancy over time, realized ephemeral fraction.

## Predictions (registered before the run)

- The lifecycle / economic policy shows **no significant win on Scenario 5**
  (uniform chat control) over cost-aware or even LRU.
- TBD: remaining per-scenario predictions — add before running.
