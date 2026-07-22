"""The command-line interface: a thin wrapper over the library functions, so
the same code serves `pip install agentic-kv-bench && agentic-kv-bench run` and
a researcher importing the package into a notebook. No logic lives here.

    agentic-kv-bench convert <corpus-dir> -o traces.jsonl
    agentic-kv-bench run <corpus-dir> --policy mymod:MyPolicy --capacity-tokens N
    agentic-kv-bench oracle <corpus-dir> --capacity-tokens N

`run` scores any policy given as import-path `module:ClassName`, so a user's
algorithm plugs in with no framework changes. That is the adoption surface.
"""

import argparse
import importlib
import json
import pathlib
import sys

from agentic_kv_bench.access import access_from_source
from agentic_kv_bench.convert import SubagentTrace, convert_trace
from agentic_kv_bench.harness import CostParams, HintDelivery, interleave, replay
from agentic_kv_bench.oracle import oracle_run, percent_of_oracle
from agentic_kv_bench.policy import Policy


def load_policy(spec: str) -> type[Policy]:
    """Resolve `module:ClassName` to a Policy subclass. The one reflection
    point, kept explicit and validated so a bad spec fails with a clear message."""
    if ":" not in spec:
        raise SystemExit(f"--policy must be 'module:ClassName', got {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise SystemExit(f"cannot import policy module {mod_name!r}: {e}") from e
    cls = getattr(mod, cls_name, None)
    if cls is None or not (isinstance(cls, type) and issubclass(cls, Policy)):
        raise SystemExit(f"{spec!r} is not a Policy subclass")
    return cls


def parse_policy_args(pairs) -> dict:
    """Turn repeated --policy-arg NAME=VALUE into constructor kwargs, so a
    parameterized policy (e.g. WA-LRU's alpha/beta/gamma) is configurable and
    every swept row is reproducible from the exact command line. Values coerce
    int -> float -> str, in that order, so '0', '0.5', and 'lru' all do the
    right thing without the caller quoting types."""
    kwargs: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--policy-arg must be NAME=VALUE, got {pair!r}")
        name, _, raw = pair.partition("=")
        for coerce in (int, float):
            try:
                kwargs[name] = coerce(raw)
                break
            except ValueError:
                continue
        else:
            kwargs[name] = raw
    return kwargs


def iter_traces(corpus: pathlib.Path):
    """Yield (path, trace-dict) for every *.json in a corpus directory or a
    single file."""
    paths = [corpus] if corpus.is_file() else sorted(corpus.glob("*.json"))
    if not paths:
        raise SystemExit(f"no .json traces found at {corpus}")
    for p in paths:
        yield p, json.loads(p.read_text())


def cmd_convert(args) -> None:
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    converted = skipped = 0
    with out.open("w") as f:
        for path, trace in iter_traces(pathlib.Path(args.corpus)):
            try:
                requests, stats = convert_trace(trace)
            except SubagentTrace:
                skipped += 1
                print(f"  defer {path.name}: subagent trace", file=sys.stderr)
                continue
            for r in requests:
                f.write(r.to_json() + "\n")
            converted += 1
    print(f"converted {converted} traces -> {out} ({skipped} subagent traces deferred)")


def _load_sessions(corpus: pathlib.Path, gap_ms: int, sim_block_tokens: int):
    """Adapt every trace and interleave them onto one timeline (real
    cross-session pressure). Returns (merged accesses, n_sessions, n_deferred)."""
    sessions, deferred = [], 0
    for _path, trace in iter_traces(corpus):
        try:
            sessions.append(access_from_source(trace, sim_block_tokens))
        except SubagentTrace:
            deferred += 1
    if not sessions:
        raise SystemExit("no replayable (non-subagent) traces in corpus")
    return interleave(sessions, gap_ms=gap_ms), len(sessions), deferred


def _check_capacity(merged, capacity_tokens: int) -> None:
    """A single request's whole prefix must fit (its attention needs all of it
    resident at once). Refuse cleanly, not with a traceback, if it does not,
    and report the minimum feasible capacity."""
    largest = max(sum(b.size_tokens for b in r.blocks) for r in merged)
    if capacity_tokens < largest:
        raise SystemExit(
            f"--capacity-tokens {capacity_tokens} is below the largest single "
            f"request's working set ({largest} tokens), which must fit at once. "
            f"Use at least {largest}. (Pressure comes from concurrent sessions "
            f"exceeding capacity, not one request exceeding it.)"
        )


def _hint_delivery(args) -> tuple[HintDelivery, str]:
    """Build the hint-degradation config from the run flags, and a short label
    for the run header. The switch positions (docs/hint-interface.md): default
    is on; --no-hints is off; --hint-delay-ms and --hint-drop-prob degrade the
    channel and compose."""
    if args.no_hints:
        return HintDelivery(enabled=False), "off"
    hints = HintDelivery(
        enabled=True, delay_ms=args.hint_delay_ms,
        drop_prob=args.hint_drop_prob, seed=args.hint_seed,
    )
    parts = []
    if args.hint_delay_ms:
        parts.append(f"delayed {args.hint_delay_ms}ms")
    if args.hint_drop_prob:
        parts.append(f"dropped p={args.hint_drop_prob} (seed {args.hint_seed})")
    return hints, ", ".join(parts) if parts else "on"


def cmd_run(args) -> None:
    policy_cls = load_policy(args.policy)
    policy_kwargs = parse_policy_args(args.policy_arg)
    kind_cost = parse_policy_args(args.kind_cost) or None
    cost = CostParams(recompute_ms_per_token=args.recompute_ms_per_token,
                      kind_cost_multiplier=kind_cost)
    hints, hint_label = _hint_delivery(args)
    merged, n_sessions, deferred = _load_sessions(
        pathlib.Path(args.corpus), args.session_gap_ms, args.sim_block_tokens
    )
    _check_capacity(merged, args.capacity_tokens)
    res = replay(merged, policy_cls(**policy_kwargs), cost, args.capacity_tokens,
                 hints=hints)
    ora = oracle_run(merged, cost, args.capacity_tokens)
    pct = percent_of_oracle(res, ora)
    pct_s = "inf" if pct == float("inf") else f"{pct:.1f}"
    print(f"policy:   {args.policy}")
    print(f"corpus:   {n_sessions} sessions interleaved "
          f"(gap {args.session_gap_ms} ms), {deferred} deferred")
    print(f"capacity: {args.capacity_tokens} tokens   "
          f"recompute: {args.recompute_ms_per_token} ms/tok   "
          f"hints: {hint_label}")
    print(f"hit rate:            {100 * res.hit_rate:.1f}%")
    print(f"scored recompute:    {res.scored_recompute_cost:.1f} "
          f"({res.scored_recompute_tokens} tokens, {res.n_evictions} evictions)")
    print(f"oracle scored cost:  {ora.scored_recompute_cost:.1f}")
    print(f"PERCENT OF ORACLE:   {pct_s}   (100 = matched the offline optimum)")


def cmd_oracle(args) -> None:
    kind_cost = parse_policy_args(args.kind_cost) or None
    cost = CostParams(recompute_ms_per_token=args.recompute_ms_per_token,
                      kind_cost_multiplier=kind_cost)
    merged, n_sessions, deferred = _load_sessions(
        pathlib.Path(args.corpus), args.session_gap_ms, args.sim_block_tokens
    )
    _check_capacity(merged, args.capacity_tokens)
    ora = oracle_run(merged, cost, args.capacity_tokens)
    print(f"{n_sessions} sessions, capacity {args.capacity_tokens} tokens: "
          f"oracle scored cost {ora.scored_recompute_cost:.1f} "
          f"({ora.capacity_misses} capacity misses, {ora.n_evictions} evictions)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentic-kv-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("convert", help="convert a kv-cache-tester corpus to the schema")
    pc.add_argument("corpus", help="a trace file or a directory of *.json traces")
    pc.add_argument("-o", "--out", required=True, help="output JSONL path")
    pc.set_defaults(func=cmd_convert)

    def add_run_args(sp):
        sp.add_argument("corpus")
        sp.add_argument("--capacity-tokens", type=int, required=True,
                        help="KV cache budget in tokens (shared across sessions)")
        sp.add_argument("--recompute-ms-per-token", type=float, default=1.0,
                        help="cost model parameter, swept per the Phase 2 verdict")
        sp.add_argument("--session-gap-ms", type=int, default=1000,
                        help="v1 arrival overlay: stagger between session starts")
        sp.add_argument("--sim-block-tokens", type=int, default=256,
                        help="simulated block granularity (sweep default 256)")
        sp.add_argument("--kind-cost", action="append", metavar="KIND=MULT",
                        help="per-kind recompute-cost multiplier (repeatable), e.g. "
                             "--kind-cost tool_output=0.2 --kind-cost reasoning=1.0; "
                             "omitted => uniform cost")

    pr = sub.add_parser("run", help="replay a policy against a corpus, report percent-of-oracle")
    add_run_args(pr)
    pr.add_argument("--policy", required=True, help="import path 'module:ClassName'")
    pr.add_argument("--policy-arg", action="append", metavar="NAME=VALUE",
                    help="constructor kwarg for a parameterized policy "
                         "(repeatable), e.g. --policy-arg alpha=1 --policy-arg beta=0")
    pr.add_argument("--no-hints", action="store_true",
                    help="hints off: the inference-only degradation position")
    pr.add_argument("--hint-delay-ms", type=int, default=0,
                    help="hint degradation: deliver each lifecycle hint N ms late")
    pr.add_argument("--hint-drop-prob", type=float, default=0.0,
                    help="hint degradation: drop each hint with probability p")
    pr.add_argument("--hint-seed", type=int, default=0,
                    help="seed for the (reproducible) hint-drop channel")
    pr.set_defaults(func=cmd_run)

    po = sub.add_parser("oracle", help="report the oracle's cost for each trace")
    add_run_args(po)
    po.set_defaults(func=cmd_oracle)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
