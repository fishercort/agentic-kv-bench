"""SimPort and the Port-driven simulator.

`SimPort` implements the neutral `Port` seam (kv_policy_core) over the simulator's
resident set. `replay_via_port` is the simulator driven entirely through that seam:
same scoring as `harness.replay`, but every admission and eviction goes through the
Port. In step 2 the same `replay_via_port` runs with a `MiniservePort` over the real
block pool, and the parity harness (`assert_parity`) asserts the two agree — that is
the offline/online-parity check the whole thesis rests on.

For step 1 there is one Port (SimPort), so the parity harness compares
`replay_via_port` against the reference `harness.replay`: if the Port-driven path
reproduces the reference exactly on the traces, the seam is faithful.
"""

from agentic_kv_bench.harness import (
    _HINTS_ON,
    CostParams,
    HintDelivery,
    RequestAccess,
    RunResult,
    _build_hint_schedule,
)
from kv_policy_core import BlockMeta, CacheView, Policy


class SimPort:
    """The simulator's implementation of the Port seam. Owns the resident set and
    the eviction mechanics; the policy is driven against it exactly as it will be
    driven against MiniservePort. `mode` is enforce (decisions enacted) or advisory
    (recorded but not enacted) — the shadow-before-enforce switch."""

    def __init__(self, capacity_tokens: int, mode: str = "enforce"):
        self.capacity_tokens = capacity_tokens
        self.mode = mode
        self._resident: dict = {}
        self.view = CacheView(self._resident)
        self.resident_tokens = 0
        self.evictions = 0
        self._policy: Policy | None = None

    def bind_policy(self, policy: Policy) -> None:
        self._policy = policy
        policy.bind(self.view)

    def set_protected(self, working: frozenset) -> None:
        self.view._set_protected(working)

    def resident(self) -> dict:
        return self._resident

    def get(self, block_id):
        return self._resident.get(block_id)

    def admit(self, meta: BlockMeta) -> None:
        self._resident[meta.block_id] = meta
        self.resident_tokens += meta.size_tokens

    def _reap(self, vid) -> None:
        m = self._resident.pop(vid)
        self.resident_tokens -= m.size_tokens
        self.evictions += 1

    def reap_if_resident(self, vid, protected: frozenset) -> None:
        """Proactive (maintain) reap: only if resident and not currently needed."""
        if vid in self._resident and vid not in protected:
            self._reap(vid)

    def free_to_fit(self, needed_tokens: int, now_ms: int, protected: frozenset) -> None:
        while self.capacity_tokens - self.resident_tokens < needed_tokens:
            evictable = sum(
                m.size_tokens for b, m in self._resident.items() if b not in protected
            )
            if evictable < needed_tokens - (self.capacity_tokens - self.resident_tokens):
                raise RuntimeError(
                    "request working set exceeds capacity: its prefix cannot be "
                    "held resident even after evicting everything evictable"
                )
            victims = self._policy.evict(
                needed_tokens - (self.capacity_tokens - self.resident_tokens), now_ms
            )
            if not victims:
                raise RuntimeError(
                    "policy.evict returned no victims but space is needed; "
                    "a correct policy must free space or the request cannot be admitted"
                )
            for vid in victims:
                if vid in protected or vid not in self._resident:
                    raise RuntimeError(
                        f"policy tried to evict block {vid}, which is "
                        f"{'needed by the current request' if vid in protected else 'not resident'}"
                    )
                self._reap(vid)

    def now_ms(self) -> int:  # sim time comes from the trace; unused by the seam
        return 0


def replay_via_port(
    accesses: list[RequestAccess],
    policy: Policy,
    cost: CostParams,
    capacity_tokens: int,
    hints: HintDelivery | None = None,
    high_value_threshold: float | None = None,
) -> RunResult:
    """The simulator driven through the Port seam. Scoring identical to
    harness.replay; the only difference is that admission/eviction go through
    SimPort, which is what step 2 swaps for MiniservePort."""
    hints = hints if hints is not None else _HINTS_ON
    port = SimPort(capacity_tokens)
    port.bind_policy(policy)
    seen: set = set()

    hits = compulsory = capacity = hv_capacity = 0
    scored_cost = 0.0
    scored_tokens = 0
    peak = 0
    schedule = _build_hint_schedule(accesses, hints)
    hp = 0

    for req in accesses:
        now = req.arrival_ms
        working = frozenset(b.block_id for b in req.blocks)
        port.set_protected(working)
        while hp < len(schedule) and schedule[hp][0] <= now:
            policy.on_hint(schedule[hp][1], schedule[hp][0])
            hp += 1
        for vid in policy.maintain(now):
            port.reap_if_resident(vid, working)
        for bref in req.blocks:
            m = port.get(bref.block_id)
            if m is not None:  # hit
                hits += 1
                m.last_access_ms = now
                m.access_count += 1
                policy.on_access(bref.block_id, m, now)
                continue
            rc = cost.recompute_cost(bref.size_tokens, bref.kind)
            first_time = bref.block_id not in seen
            if first_time:
                compulsory += 1
                seen.add(bref.block_id)
            else:
                capacity += 1
                scored_cost += rc
                scored_tokens += bref.size_tokens
                if high_value_threshold is not None and rc >= high_value_threshold:
                    hv_capacity += 1
            if bref.size_tokens > capacity_tokens:
                raise RuntimeError(
                    f"block {bref.block_id} ({bref.size_tokens} tok) exceeds "
                    f"capacity ({capacity_tokens} tok); no policy can admit it"
                )
            port.free_to_fit(bref.size_tokens, now, working)
            meta = BlockMeta(
                block_id=bref.block_id, kind=bref.kind,
                size_tokens=bref.size_tokens, recompute_cost=rc,
                last_access_ms=now, access_count=1,
            )
            meta.tier = policy.place(meta)
            port.admit(meta)
            policy.on_access(bref.block_id, meta, now)
            peak = max(peak, port.resident_tokens)

    while hp < len(schedule):
        policy.on_hint(schedule[hp][1], schedule[hp][0])
        hp += 1

    return RunResult(
        total_accesses=hits + compulsory + capacity,
        hits=hits,
        compulsory_misses=compulsory,
        capacity_misses=capacity,
        scored_recompute_cost=scored_cost,
        scored_recompute_tokens=scored_tokens,
        high_value_capacity_misses=hv_capacity,
        peak_resident_tokens=peak,
        n_evictions=port.evictions,
    )
