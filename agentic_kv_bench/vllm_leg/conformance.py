"""Conformance fixtures for the telemetry Compute contract, generated from the Python
reference agent. Any implementation (including the Rust `infertap` agent) must reproduce
`expect` given `config` + `events`. Fixtures live in the public spec repo
(kv-policy-core/conformance/telemetry/) so every implementation checks the same versioned
data — the golden reference that makes the Rust port a bounded, correctness-checked build.

Portability: block hashes in fixtures are integers, and the salted `block_id` is
`sha256(salt || 0x00 || domain || 0x00 || decimal(hash))[:16]` — `decimal(int)` is
language-neutral (Python `repr(int) == str(int)`), so Rust reproduces it exactly. A non-int
hash needs a canonical encoding defined before it belongs in a fixture.

Scope: the OPEN module contract — normalization, resident/evicted tracking, residual and
eviction DETECTION, catalog stats, metadata-only records. (Dollarization / band economics
are pipeline consumers, not part of this contract.)
"""

import json
from pathlib import Path

from .agent import TelemetryAgent
from .event_model import ClearedAll, RemovedBlock, StoredBlock

FIXTURE_VERSION = "v0"
_PROV = {"gpu": "fixture", "vllm_version": "0.11.0", "event_schema_version": "vllm-0.11.0-1"}
_SALT = "conformance-salt"


def _s(source, ts, h, block_size=4, medium="GPU"):
    return {"source": source, "type": "StoredBlock", "ts": ts, "block_hash": h,
            "block_size": block_size, "parent_hash": None, "medium": medium}


def _r(source, ts, h, medium="GPU"):
    return {"source": source, "type": "RemovedBlock", "ts": ts, "block_hash": h,
            "medium": medium}


def _c(source, ts, medium="GPU"):
    return {"source": source, "type": "ClearedAll", "ts": ts, "medium": medium}


def _event(d):
    t = d["type"]
    if t == "StoredBlock":
        return StoredBlock(d["block_hash"], d.get("parent_hash"), d["block_size"],
                           d.get("medium"), d["ts"])
    if t == "RemovedBlock":
        return RemovedBlock(d["block_hash"], d.get("medium"), d["ts"])
    if t == "ClearedAll":
        return ClearedAll(d.get("medium"), d["ts"])
    raise ValueError(t)


def run(config, events):
    """Run the reference agent over a fixture's events; return its `expect` block."""
    agent = TelemetryAgent(_PROV, salt=_SALT, block_size=config["block_size"],
                           evicted_cap=config.get("evicted_cap"))
    for ev in events:
        agent.ingest(ev["source"], _event(ev))
    st = agent.catalog_stats()
    return {
        "residual_tokens": agent.residual_tokens,
        "eviction_recompute_tokens": agent.eviction_recompute_tokens,
        "catalog_peak_entries": st["catalog_peak_entries"],
        "evicted_aged_out": st["evicted_aged_out"],
        "record_counts": st["record_counts"],
        "records": list(agent.records),
    }


def _scenarios():
    return [
        {"name": "cross_instance_residual", "config": {"block_size": 4}, "events": [
            _s("a", 1.0, 1), _s("a", 1.1, 2), _s("a", 1.2, 3),
            _s("b", 2.0, 1), _s("b", 2.1, 2), _s("b", 2.2, 9)]},  # 1,2 on a -> residual 8
        {"name": "no_false_sharing", "config": {"block_size": 4}, "events": [
            _s("a", 1.0, 1), _s("b", 2.0, 2)]},                   # disjoint -> residual 0
        {"name": "eviction_recompute", "config": {"block_size": 4}, "events": [
            _s("a", 1.0, 5), _r("a", 2.0, 5), _s("a", 3.0, 5)]},  # re-store after evict -> 4
        {"name": "lifecycle_clear", "config": {"block_size": 4}, "events": [
            _s("a", 1.0, 1), _s("a", 1.1, 2), _c("a", 2.0)]},     # 2 clear lifecycle records
        {"name": "evicted_cap_ageout", "config": {"block_size": 4, "evicted_cap": 2},
         "events": [_s("a", 1.0, 1), _r("a", 1.1, 1), _s("a", 2.0, 2), _r("a", 2.1, 2),
                    _s("a", 3.0, 3), _r("a", 3.1, 3),            # cap 2 -> block 1 aged out
                    _s("a", 4.0, 1), _s("a", 5.0, 3)]},          # 1 not counted, 3 -> 4
    ]


def build():
    return [{"fixture_version": FIXTURE_VERSION, "salt": _SALT, "provenance": _PROV,
             **s, "expect": run(s["config"], s["events"])} for s in _scenarios()]


def conformance_dir():
    """The public home of the fixtures: kv-policy-core/conformance/telemetry/ (resolved via
    the installed kv_policy_core package)."""
    import kv_policy_core
    return Path(kv_policy_core.__file__).resolve().parent.parent / "conformance" / "telemetry"


def write():
    d = conformance_dir()
    d.mkdir(parents=True, exist_ok=True)
    for fx in build():
        (d / f"{fx['name']}.json").write_text(json.dumps(fx, indent=1) + "\n")
    return d


if __name__ == "__main__":
    print("wrote fixtures to", write())
