# The lifecycle hint interface

The hint interface is how out-of-band lifecycle knowledge (information no
access-pattern statistic contains) reaches a policy. The signal-blind ladder
(LRU, IdleTTL, GDSF, WA-LRU) showed the cheap statistics collect under 10% of
LRU's room over the oracle; the remaining points need to know which blocks are
*dead*, and that is what a hint carries.

An interface is only real if it degrades gracefully when the hint is missing,
late, or lossy. This file is the contract and its failure modes.

## The event contract

A hint is a plain `dict` delivered to `Policy.on_hint(event, now_ms)`. Events
travel on `RequestAccess.lifecycle_events` and are opaque to the harness; only
policies interpret them. Every event has:

- `event`: the type string.
- `at_ms`: when the event happened in trace time (its delivery time before any
  degradation).

The type wired end-to-end on the current corpus:

- `retire` — the named blocks are dead (will never be accessed again) and may be
  reclaimed. Payload: `block_ids` (a list of cache block_ids, in the exact id
  space `cache.resident()` uses). Emitted at each **compaction**: the serving
  framework performed the compaction, so it knows precisely which KV it dropped.
  This models the retired-cache paper's termination message (arXiv 2605.06472):
  the message *is* the hint.

`block_ids` are minted in `access.py` (session-namespaced, coarsened per the
simulation's block granularity), not in the portable schema, because a hint is
actionable only if it names blocks in the id space the policy sees. The schema
keeps the compaction as a portable count; the execution form resolves it into
named retirements. That is the one deliberate divergence between the two.

### Truth guarantee

A `retire` hint is always **true**: a block it names is verified (via ground-
truth last-use) never to be accessed again. A policy that trusts the hint is
never punished for a correct signal. All imperfection is injected explicitly by
the degradation switches below, never by the emitter lying about deadness. (A
future `corruption` switch could model wrong hints; the spec's switches are
delay and drop, so that is out of scope here and would be named if added.)

## Delivery and timing

The harness builds a delivery schedule from all events, then delivers each hint
the first time the replay clock reaches its (possibly delayed) delivery time,
**before** `maintain()` in that step, so a lifecycle policy can reclaim freshly
retired blocks in the same request. `now_ms` passed to `on_hint` is the delivery
time (when the policy observes the hint), which the policy uses as its clock.

## Degradation switches (the robustness curve)

`HintDelivery(enabled, delay_ms, drop_prob, seed)` sets the channel per run. The
four positions the spec names:

| Position | Config | Meaning |
|----------|--------|---------|
| on | `HintDelivery()` | every hint, at its event time |
| off | `HintDelivery(enabled=False)` | policy sees no hints (inference-only) |
| delayed | `HintDelivery(delay_ms=N)` | every hint arrives N ms late |
| dropped | `HintDelivery(drop_prob=p, seed=s)` | each hint lost with probability p |

Delay and drop **compose** (a channel can be both late and lossy). Drop is
seeded, so a lossy run reproduces exactly. From the CLI: `--no-hints`,
`--hint-delay-ms N`, `--hint-drop-prob p`, `--hint-seed s`.

## The graceful-degradation contract

A hint-consuming policy must degrade to a signal-blind policy as hints vanish.
`RetiredCache` is the reference: with hints off (or fully dropped) no block is
ever marked retired, so it is **exactly LRU**. The with-hints vs hints-off gap
is therefore precisely what the retirement signal buys, and the delay/drop sweep
between them is the robustness curve that makes the interface a credible contract
rather than a demo assumption. The headline lifecycle comparison the project was
built toward is exactly this pair: retired-cache with hints vs inference-only.

## Failure modes, enumerated

- **Hints off / all dropped** → policy == its signal-blind fallback (contract).
- **Hints delayed** → reclamation happens late; the block occupies capacity
  longer, so the benefit shrinks toward the fallback as delay grows.
- **Hints partially dropped** → a fraction of dead blocks are never reclaimed;
  benefit interpolates between with-hints and off along the drop probability.
- **Hint after the last access** → delivered for contract completeness but cannot
  change scoring (nothing follows to reclaim for).
