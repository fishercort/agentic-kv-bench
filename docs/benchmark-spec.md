# agentic-kv-bench — benchmark specification

## The gap

Agentic workloads changed the shape of KV cache — subagents spawn and die,
reasoning blocks open and close, tool outputs go cold instantly. Four research
groups converged on agentic KV cache management within eight months. CacheTTL
(arXiv 2511.02230, first released as Continuum) came first from the systems
side, proposing KV TTL to keep
cache alive through tool-call gaps. In May–June 2026 two papers attacked the
block-level gap directly: prediction-based lifecycle-aware eviction (arXiv
2605.06472, retired-cache reclamation via workflow termination messages) and
SAGA's Workflow-Aware LRU (arXiv 2605.00528). IntentKV (arXiv 2606.09916)
attacked the same agentic problem from the token-level lane (see Scope below).
Each is evaluated on its own private setup. There is no common benchmark, no
shared baseline discipline, no oracle-relative reporting, and none of it is
merged into vLLM, LMCache, or Dynamo mainline.

The block-level lane alone now has three competing answers — TTL,
prediction-based reclamation, workflow-aware scoring — and no ground truth.
This benchmark builds the ground truth — measured costs, a common benchmark, an
oracle — so policies can be compared on the same traces under the same
discipline.

## Scope: two lanes

There are two distinct lanes of KV cache reduction, and this benchmark is about
exactly one of them:

- **Token-level eviction** (SnapKV / H2O lineage) — lossy: it drops individual
  tokens' KV and trades model accuracy for memory. An ML problem, and it already
  has benchmarks.
- **Block/request-level management** (this benchmark's lane) — lossless: whole
  blocks are evicted, offloaded, or recomputed, and the exact KV is always
  reconstructable. A systems problem, and it has no common benchmark.

This benchmark measures the block/request-level lane only. The distinction is
both the scoping guard (token-level methods are related work, not baselines
here) and the gap statement (the lossless lane is the one without shared
evaluation).

## Governing constraint: defensibility

A skeptical reviewer must not be able to say the trace distributions were
invented to flatter the evictor. Every design choice in the trace suite serves that
constraint, through three defenses: anchor to public data, expose every knob,
and include workloads where the policy should lose (see Scenario 5, the control,
in `trace-schema.md`).

## Related work and methodology alignment

Where this sits, and what it deliberately borrows so results are comparable
rather than novel-for-novelty's-sake.

- **The field, mapped.** A 2026 survey on system-aware KV-cache optimization
  (arXiv 2607.08057) is the related-work map. The block/request-level lane this
  benchmark measures holds at least four converging methods (CacheTTL/Continuum
  2511.02230, retired-cache 2605.06472, SAGA WA-LRU 2605.00528, and others);
  the token-level lane (IntentKV 2606.09916, l2-norm eviction, learned per-head
  methods such as KVP) is related work, not baselines here, per Scope. The
  field is still growing and still has no common evaluation, which is the gap.
- **Serving-metric vocabulary.** Reported metrics follow vLLM's
  `benchmark_serving` convention (TTFT and TPOT as mean/median/P99, throughput,
  goodput, over an arrival-rate sweep), so a reader can place these numbers
  beside a vLLM or LMCache benchmark without translation. Deviating from that
  vocabulary would make the benchmark harder to adopt, which defeats its
  purpose.
- **Arrival and workload calibration.** Arrival processes calibrate to the
  public Azure LLM-inference conversation traces and Mooncake production
  traces; agentic turn and tool-call structure is anchored to the recognized
  agentic workloads (SWE-bench trajectories, BFCL) and complemented by this
  benchmark's real Claude Code corpus, which most prior work lacks (it
  evaluates on synthetic RULER/GSM8K or older ShareGPT/OASST2).
- **Oracle grounding.** The offline oracle is grounded in caching theory
  (Belady's MIN and the NP-hardness of general variable-size caching); see
  `oracle.md`. The point is a principled approximation with a named gap, not an
  unfalsifiable "optimal."

## Design-for-adoption requirements

An artifact researchers and practitioners can pull and use, independent of any
one policy. Each requirement is cheap individually; jointly they separate "used"
from "starred and forgotten":

- **pip-installable, zero-friction repro:** `pip install agentic-kv-bench`, then
  one command replays every canonical trace against every bundled policy and
  emits the results table. If reproducing the README takes more than five
  minutes of setup, adoption dies.
- **Real traces first:** anchor on the public
  [kv-cache-tester corpus](https://github.com/callanjfox/kv-cache-tester) (739
  anonymized Claude Code conversation traces, already used by
  [LMCache's May 2026 MI300X benchmark](https://blog.lmcache.ai/en/2026/05/12/benchmarking-lmcache-for-multi-turn-agentic-workloads-on-amd-mi300x/))
  — cite and credit, don't re-collect. The synthetic generator
  exists for the one thing real traces can't give: a controllable
  ephemeral-fraction sweep and scenario isolation.
- **Bundled baselines:** LRU, GDSF-style cost-aware, WA-LRU (SAGA), and
  retired-cache lifecycle eviction (2605.06472) ship in the box. A researcher
  comparing against "the field" should need zero reimplementation.
- **The oracle in the box:** a Belady-style approximate oracle so every policy —
  theirs included — reports percent-of-oracle out of the gate.
- **A stable policy interface:** `on_access / on_hint / evict / place`, small
  enough to implement in an afternoon, documented with a worked example.
- **A leaderboard-style README:** the current results table, wins AND losses,
  with the pre-registered EVAL_PLAN.md linked.
- **Apache-2.0, semver, canonical pinned traces** so results are comparable
  across papers that cite different versions.
- **Portable cost model — `agentic-kv-bench calibrate`:** ship the cost-model
  measurement scripts as a CLI that measures prefill curves and tier bandwidths
  on the user's own hardware/model and emits a cost-model config the simulator
  consumes. Bundled constants are the default; calibrated constants make
  deployment-decision benching credible on the user's own stack. This is the
  second life of the measurement methodology — a reusable tool, not a one-off
  writeup. Because the cost model is parametric, conclusions are reported
  across model-scale cost profiles — anchored profiles carry measured claims,
  derived profiles are labeled extrapolations — see EVAL_PLAN.md.
- **Proprietary-workload path, documented:** a short guide for converting private
  production logs into the trace schema locally. Nothing phones home; the
  enterprise use case ("which policy should we deploy, and how far from the
  oracle ceiling are we?") runs entirely on the user's machines.

## Build order

Suggested construction order for the benchmark:

1. Schema + replayer + LRU baseline in simulator mode.
2. Scenario 5 control + Scenario 2 fan-out (the two extremes).
3. Oracle.
4. Remaining scenarios + calibration pass against public data.
5. Hint-degradation switches + EVAL_PLAN.md pre-registration.

## Packaging as the standalone artifact

A separate repo named as a benchmark: generator + canonical traces + oracle +
harness + policy interface + baseline implementations (LRU, GDSF-style
cost-aware). Its README describes the missing benchmark and invites policies —
it does not lead with any one policy. The cost-model-driven joint policy (see
`policy-interface.md`) is then the first submission, not the reason the benchmark
exists.
