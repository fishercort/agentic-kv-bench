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

- Ephemeral fraction: 0.1 → 0.7  (grid points: TBD — pinned and dated before
  the first full sweep)
- Memory-pressure levels: TBD — pinned and dated before the first full sweep
- Cost profiles (model scale): the cost model is parametric, so the full policy
  comparison runs under multiple model-scale cost profiles without serving
  those models. **Anchored** (measured via the calibrate CLI on served
  hardware): 1.5B on miniserve; 8B-class in the production-engine validation
  run. **Budgeted option, executed schedule-permitting:** a 70B-class anchor on
  a rented 2×H100 TP2 vLLM deployment — calibrate + full trace sweep, first
  thing cut under time pressure. **Unanchored** profiles (reasoning-heavy; 70B
  if the budgeted run is cut) are derived from config arithmetic plus scaling
  assumptions documented in the profile files; conclusions at derived scales
  are reported as extrapolations, with anchored scales carrying the measured
  claims.
  - 70B confound, named: at TP2, KV blocks span two GPUs and the cost model
    gains an NVLink transfer term absent at single-GPU scales — a labeled
    difference in the profile, not a silent one. The same deployment yields a
    near-free second profile: 70B at fp8 halves bytes/token with one flag —
    run it if the rental happens.
- Hint modes: on / delayed by N ms / dropped w.p. p / off
- Policies: LRU, CacheTTL (Continuum) TTL, GDSF-style cost-aware, WA-LRU, retired-cache
  lifecycle, economic joint
- Cross-product, stated explicitly: hint-consuming policies (retired-cache
  lifecycle, economic joint) run **with-hints and inference-only as separate
  rows** — that split is a headline ablation, not a footnote. The full grid is
  policy × hint-mode × pressure × ephemeral-fraction × cost-profile; cells
  skipped from the full cross-product are enumerated with reasons here before
  the sweep runs (TBD — pinned and dated before the first full sweep).
- Seeds: TBD — pinned and dated before the first full sweep

## Metrics

Percent-of-oracle (headline), tokens recomputed, high-value hit rate, p95/p99
TTFT (live mode), KV occupancy over time, realized ephemeral fraction.

Policy ranking and effect size are reported **as a function of model-scale cost
profile**, not at a single scale: small-model economics make recompute nearly
free and can shrink effect sizes to artifacts, so a single-scale result — at
any scale — is one point where a curve is required. The registered question:
at what model scale does lifecycle-aware eviction start paying for itself?

## Predictions (registered before the run)

- The lifecycle / economic policy shows **no significant win on Scenario 5**
  (uniform chat control) over cost-aware or even LRU.
- TBD: remaining per-scenario predictions — add before running.
