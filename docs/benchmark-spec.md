# agentic-kv-bench — benchmark specification

## The gap

Agentic workloads changed the shape of KV cache — subagents spawn and die,
reasoning blocks open and close, tool outputs go cold instantly. In May–June
2026 three concurrent papers attacked exactly this gap: prediction-based
lifecycle-aware eviction (arXiv 2605.06472, retired-cache reclamation via
workflow termination messages), SAGA's Workflow-Aware LRU (arXiv 2605.00528),
and IntentKV (arXiv 2606.09916). Each is evaluated on its own private setup.
There is no common benchmark, no shared baseline discipline, no oracle-relative
reporting, and none of it is merged into vLLM, LMCache, or Dynamo mainline.

The field produced three competing answers and no ground truth. This benchmark
builds the ground truth — measured costs, a common benchmark, an oracle — so
policies can be compared on the same traces under the same discipline.

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

A skeptical reviewer must not be able to say "you invented a distribution that
flatters your evictor." Every design choice in the trace suite serves that
constraint, through three defenses: anchor to public data, expose every knob,
and include workloads where the policy should lose (see Scenario 5, the control,
in `trace-schema.md`).

## Design-for-adoption requirements

An artifact researchers and practitioners can pull and use, independent of any
one policy. Each requirement is cheap individually; jointly they separate "used"
from "starred and forgotten":

- **pip-installable, zero-friction repro:** `pip install agentic-kv-bench`, then
  one command replays every canonical trace against every bundled policy and
  emits the results table. If reproducing the README takes more than five
  minutes of setup, adoption dies.
- **Real traces first:** anchor on the public kv-cache-tester corpus (739
  anonymized Claude Code conversation traces, already used by LMCache's May 2026
  MI300X benchmark) — cite and credit, don't re-collect. The synthetic generator
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
- **Portable cost model — `agentic-kv-bench calibrate`:** ship the Phase 2
  measurement scripts as a CLI that measures prefill curves and tier bandwidths
  on the user's own hardware/model and emits a cost-model config the simulator
  consumes. Bundled constants are the default; calibrated constants make
  deployment-decision benching credible on the user's own stack. This is the
  second life of the Phase 2 measurement methodology — a reusable tool, not a
  one-off writeup.
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
