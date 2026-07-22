"""The policy interface — now hosted by the neutral `kv_policy_core` package.

The canonical definitions moved to `kv_policy_core` so that the engines (miniserve,
later vLLM/LMCache adapters) can depend on the interface without depending on the
benchmark (see docs/L2-integration-design.md). This module re-exports them so
existing `from agentic_kv_bench.policy import ...` imports keep working unchanged.
"""

from kv_policy_core import BlockMeta, CacheView, Decision, Evict, Policy, Port

__all__ = ["BlockMeta", "CacheView", "Decision", "Evict", "Policy", "Port"]
