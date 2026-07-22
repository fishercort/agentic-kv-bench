"""The offline oracle: cost-aware Belady with lookahead (docs/oracle.md).

Belady's MIN evicts the block used farthest in the future; it is optimal only
for uniform cost and size. This benchmark has neither (variable recompute cost,
variable span size), and the variable-size case is NP-hard offline, so this is
a strong APPROXIMATION, not a provable optimum. Named as such: percent-of-oracle
is percent-of-this-strong-offline-baseline. Every online policy is measured
against it because it uses perfect future knowledge the Policy interface
deliberately denies.

The eviction rule, in priority order:
  1. Evict blocks never accessed again (evicting them costs zero future work).
  2. Otherwise evict the block whose next use is farthest away (Belady),
     tie-broken by lowest recompute cost (cheapest to bring back if wrong).
Scored the same way as replay(): only capacity misses (evicted-then-reaccessed)
count, so oracle and policies are compared on identical accounting.
"""

from agentic_kv_bench.harness import CostParams, RequestAccess, RunResult

NEVER = 1 << 60  # farthest possible next use: a block accessed again never


def _next_use_index(accesses: list[RequestAccess]) -> list[dict[int, int]]:
    """future_next[i][block_id] = the earliest request index > i that accesses
    block_id, or a large sentinel if never again. Built by a single reverse
    pass so the oracle has O(1) next-use lookup during the forward replay."""
    n = len(accesses)
    future: list[dict[int, int]] = [dict() for _ in range(n)]
    last_seen: dict[int, int] = {}
    for i in range(n - 1, -1, -1):
        # next use, as of BEFORE request i, is what we recorded going backward
        future[i] = dict(last_seen)
        for b in accesses[i].blocks:
            last_seen[b.block_id] = i
    return future


def oracle_run(
    accesses: list[RequestAccess], cost: CostParams, capacity_tokens: int
) -> RunResult:
    future = _next_use_index(accesses)
    resident: dict[int, dict] = {}  # block_id -> {size, rc, kind}
    resident_tokens = 0
    seen: set[int] = set()

    hits = compulsory = capacity = evictions = hv = 0
    scored_cost = 0.0
    scored_tokens = 0
    peak = 0

    for i, req in enumerate(accesses):
        working = frozenset(b.block_id for b in req.blocks)
        for bref in req.blocks:
            if bref.block_id in resident:
                hits += 1
                continue
            rc = cost.recompute_cost(bref.size_tokens)
            if bref.block_id not in seen:
                compulsory += 1
                seen.add(bref.block_id)
            else:
                capacity += 1
                scored_cost += rc
                scored_tokens += bref.size_tokens
            if bref.size_tokens > capacity_tokens:
                raise RuntimeError(
                    f"block {bref.block_id} exceeds capacity; no policy admits it"
                )
            # Evict with future knowledge, never touching the current working
            # set (same invariant the harness enforces for online policies).
            while capacity_tokens - resident_tokens < bref.size_tokens:
                victim = _pick_victim(resident, future[i], working)
                if victim is None:
                    raise RuntimeError(
                        "request working set exceeds capacity (oracle path)"
                    )
                m = resident.pop(victim)
                resident_tokens -= m["size"]
                evictions += 1
            resident[bref.block_id] = {"size": bref.size_tokens, "rc": rc}
            resident_tokens += bref.size_tokens
            peak = max(peak, resident_tokens)

    return RunResult(
        total_accesses=hits + compulsory + capacity,
        hits=hits,
        compulsory_misses=compulsory,
        capacity_misses=capacity,
        scored_recompute_cost=scored_cost,
        scored_recompute_tokens=scored_tokens,
        high_value_capacity_misses=hv,
        peak_resident_tokens=peak,
        n_evictions=evictions,
    )


def _pick_victim(
    resident: dict[int, dict], next_use: dict[int, int], protected: frozenset[int]
) -> int | None:
    """Farthest next use wins (never-again = NEVER = farthest), tie-broken by
    lowest recompute cost. Never returns a protected (currently-needed) block;
    None if nothing is evictable."""
    best_id, best_key = None, None
    for bid, m in resident.items():
        if bid in protected:
            continue
        nu = next_use.get(bid, NEVER)
        key = (nu, -m["rc"])  # maximize next-use, then minimize rc
        if best_key is None or key > best_key:
            best_key, best_id = key, bid
    return best_id


def percent_of_oracle(policy: RunResult, oracle: RunResult) -> float:
    """The headline metric. 100 means matched the oracle; higher means the
    policy paid more avoidable recompute than the offline optimum. If the
    oracle itself pays zero (no capacity pressure), a policy that also pays
    zero is 100, and any policy cost is reported as infinite overrun."""
    if oracle.scored_recompute_cost == 0.0:
        return 100.0 if policy.scored_recompute_cost == 0.0 else float("inf")
    return 100.0 * policy.scored_recompute_cost / oracle.scored_recompute_cost
