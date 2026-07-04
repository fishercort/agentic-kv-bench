# agentic-kv-bench — the oracle

Belady-style offline replay with perfect future knowledge.

- For each eviction decision point, the oracle knows every future access and
  every future lifecycle event, and evicts to minimize total reconstruction cost
  (from the calibrated cost model), not just miss count.
- Honesty requirement, stated in the writeup: with variable block sizes and
  variable reconstruction costs this is a knapsack-flavored problem, so the
  oracle is a strong approximation (cost-greedy with lookahead), not provable
  optimality. Name the approximation; do not call it optimal.
- Output: per-trace lower-bound cost curve. Every policy result is then reported
  as **percent-of-oracle**, the headline metric.
