# agentic-kv-bench — the oracle

Belady-style offline replay with perfect future knowledge.

- For each eviction decision point, the oracle knows every future access and
  every future lifecycle event, and evicts to minimize total reconstruction cost
  (from the calibrated cost model), not just miss count.
- Honesty requirement, grounded in caching theory rather than hedged: Belady's
  MIN (1966) is offline-optimal only for uniform cost and uniform size. This
  benchmark has neither. Weighted caching (variable reconstruction cost,
  uniform size) is still solvable optimally offline, but the variable-length
  spans this corpus produces make it general caching (variable object size),
  which is NP-hard offline. The oracle is therefore a principled approximation
  (cost-greedy with lookahead), not a provable optimum, and the writeup names
  it as such and reports its own approximation gap where a bound is known.
- Output: per-trace lower-bound cost curve. Every policy result is then reported
  as **percent-of-oracle**, the headline metric.
