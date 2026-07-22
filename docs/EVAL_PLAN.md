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

Pinned 2026-07-21 (the trigger: the first end-to-end percent-of-oracle exists,
see results/control-lru-low-pressure.json). Pressure and session-count are
expressed as RATIOS, not absolute tokens, so the grid survives the block-
granularity decision below.

- Memory pressure = cache capacity / combined working set of concurrent
  sessions: {1.5, 1.0, 0.6, 0.3}. 1.5 is the low-pressure control regime (LRU
  expected near-optimal, confirmed at 100.5% on real traces); 0.3 is heavy
  pressure where lifecycle awareness should matter.
- Concurrency = number of interleaved sessions: {2, 8, 32}. Combined with the
  pressure ratio this sets the absolute capacity per cell.
- Simulated block granularity: {256} tokens as the sweep default, with a
  one-cell sensitivity check at {64, 256} confirming policy RANKINGS are
  granularity-invariant (every policy faces the same quantization). Granularity
  is a simulation parameter, not an engine constraint; coarsening from 64 to
  256 cuts resident-block counts ~4x and is the first-order performance lever.
- Ephemeral fraction: 0.1 → 0.7 in 0.15 steps (synthetic scenarios; realized
  fraction reported per real trace, not targeted).
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
- Policies: LRU (floor), IdleTTL (naive idle-time strawman, the mechanism
  lower bound — NOT Continuum), GDSF-style cost-aware, WA-LRU (SAGA),
  retired-cache lifecycle, a faithful CacheTTL/Continuum (gap-aware protection,
  hint-consuming — distinct from IdleTTL, lands with the hint interface), and
  the economic joint policy last
- Cross-product, stated explicitly: hint-consuming policies (retired-cache
  lifecycle, economic joint) run **with-hints and inference-only as separate
  rows** — that split is a headline ablation, not a footnote. The full grid is
  policy × hint-mode × pressure × ephemeral-fraction × cost-profile; cells
  skipped from the full cross-product are enumerated with reasons here before
  the sweep runs (TBD — pinned and dated before the first full sweep).
- Seeds: TBD — pinned and dated before the first full sweep

### Weight sweep pre-registration (WA-LRU; pinned 2026-07-21, before the first weighted run)

WA-LRU has tunable weights, so it is a tuning loop, and tuning loops are where
blind evaluation quietly dies: sweeping alpha/beta/gamma and reporting the best
cell is post-hoc selection unless it is labeled as the steelman it is. Structure,
fixed now:

- **Headline row = the neutral default (alpha=1, beta=1, gamma=0)**, reported as
  the primary WA-LRU number. No tuning; the honest "what the rule does out of the
  box under our estimator."
- **Tuned row = the best cell of the sweep below, reported SEPARATELY and
  labeled** "WA-LRU, weights tuned on this workload — SAGA's best case under our
  frequency estimator." This is disclosed post-hoc selection. It is reported
  because it is actually FAIRER to SAGA than the default alone: their tuned
  weights could not be extracted from the paper, so the on-workload optimum is
  the most generous signal-blind reading of their method we can give.
- Both rows ship, both labeled, tuning disclosed in the results note.

Grid (fixed before the first weighted run):
- alpha = 1 (fixed): the score is an argmax over a linear combination, so only
  the ratio alpha:beta matters; fixing alpha=1 and sweeping beta covers the space.
- beta in {0, 0.25, 0.5, 1, 2, 4}: from recency-only (beta=0) to reuse-dominated
  (beta=4). Note beta=0 is analytically identical to LRU (P_evict = alpha*R_hat,
  evict-oldest), so it is a built-in degeneracy anchor: if the optimum lands at
  beta=0, WA-LRU-tuned IS LRU and the reuse estimator added nothing.
- gamma = 0 (fixed): under uniform block size every S_hat is equal, so the gamma
  term adds the same constant to every resident block and cancels in the argmax.
  It has no effect on ranking at uniform size and is swept only when variable
  block size enters (not this workload). Stated, not silently dropped.

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
