"""Turn measured token quantities into $/month (docs/vllm-leg-runbook.md §5).

Two inputs, two roles (kept distinct, same split as spec-sheet vs achievable bandwidth):
  - provider $/GPU-hour + rate-card date: the number a prospect recognizes; goes in the
    report. Cite the rate card and date (provenance).
  - measured vLLM prefill throughput (tokens/s, fleet) on the box: the cost model's
    grounding, measured on the SAME engine whose waste we report. NOT miniserve's engine
    cost model (that feeds the separate §9 hardware corpus via gpu_day; conflating them
    would be an apples-to-oranges provenance error).

prefill $/token = (fleet $/hour) / 3600 / (fleet prefill tokens/s). A token quantity
measured over a window extrapolates to $/month by its rate times the month.
"""

SECONDS_PER_MONTH = 30 * 24 * 3600  # 2,592,000


def prefill_cost_per_token(gpu_hour_rate, n_gpus, prefill_tokens_per_sec):
    """$/token of prefill. gpu_hour_rate: provider posted $/GPU-hour (cite date).
    prefill_tokens_per_sec: measured vLLM fleet prefill throughput on the box."""
    if prefill_tokens_per_sec <= 0:
        raise ValueError("prefill_tokens_per_sec must be > 0")
    fleet_dollars_per_sec = gpu_hour_rate * n_gpus / 3600.0
    return fleet_dollars_per_sec / prefill_tokens_per_sec


def monthly_dollars(tokens, window_seconds, gpu_hour_rate, n_gpus, prefill_tokens_per_sec):
    """Extrapolate a token quantity measured over `window_seconds` to $/month of prefill.
    Used for both numbers: `tokens` = avoidable-recompute tokens (eviction waste) or
    residual tokens (modeled dedup) observed in the measurement window."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")
    cpt = prefill_cost_per_token(gpu_hour_rate, n_gpus, prefill_tokens_per_sec)
    tokens_per_month = tokens * (SECONDS_PER_MONTH / window_seconds)
    return tokens_per_month * cpt


def dollarize_result(tokens, window_seconds, prov, label):
    """Build a provenance-stamped dollarization record from a provenance dict carrying
    dollarization fields (see runbook provenance.json). Returns a dict ready to commit
    alongside the raw artifacts; `label` travels for the modeled number."""
    monthly = monthly_dollars(
        tokens, window_seconds,
        prov["gpu_hour_rate"], prov["n_gpus"], prov["prefill_tokens_per_sec"])
    return {
        "label": label,
        "tokens_measured": tokens,
        "window_seconds": window_seconds,
        "monthly_dollars": round(monthly, 2),
        "gpu_hour_rate": prov["gpu_hour_rate"],
        "rate_card_date": prov.get("rate_card_date"),
        "rate_card_url": prov.get("rate_card_url"),
        "n_gpus": prov["n_gpus"],
        "prefill_tokens_per_sec": prov["prefill_tokens_per_sec"],
        "vllm_version": prov.get("vllm_version"),
        "gpu": prov.get("gpu"),
    }
