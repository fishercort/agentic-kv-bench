"""Dollarization arithmetic (docs/vllm-leg-runbook.md §5): prefill $/token and $/month
extrapolation, with a worked hand-checkable case."""

import pytest

from agentic_kv_bench.vllm_leg.dollarize import (
    dollarize_result,
    monthly_dollars,
    prefill_cost_per_token,
)


def test_prefill_cost_per_token_hand_check():
    # $2/GPU-hr x 2 GPUs = $4/hr = $4/3600 per sec; at 40,000 tok/s -> per-token cost
    cpt = prefill_cost_per_token(gpu_hour_rate=2.0, n_gpus=2, prefill_tokens_per_sec=40_000)
    assert cpt == pytest.approx((4.0 / 3600) / 40_000)


def test_monthly_extrapolation_hand_check():
    # 1e9 avoidable-recompute tokens over a 1-hour (3600s) window, same rates.
    # tokens/month = 1e9 * (2_592_000 / 3600) = 1e9 * 720 = 7.2e11
    monthly = monthly_dollars(1e9, 3600, gpu_hour_rate=2.0, n_gpus=2,
                              prefill_tokens_per_sec=40_000)
    expected = 1e9 * (2_592_000 / 3600) * ((4.0 / 3600) / 40_000)
    assert monthly == pytest.approx(expected)


def test_zero_throughput_and_window_rejected():
    with pytest.raises(ValueError):
        prefill_cost_per_token(2.0, 2, 0)
    with pytest.raises(ValueError):
        monthly_dollars(1e9, 0, 2.0, 2, 40_000)


def test_dollarize_result_stamps_provenance_and_label():
    prov = {"gpu_hour_rate": 1.5, "n_gpus": 2, "prefill_tokens_per_sec": 50_000,
            "rate_card_date": "2026-07-22", "rate_card_url": "https://x/pricing",
            "vllm_version": "0.11.0", "gpu": "A100-80"}
    r = dollarize_result(2e8, 2700, prov, label="modeled cross-session sharing (...)")
    assert r["monthly_dollars"] > 0
    assert r["rate_card_date"] == "2026-07-22" and r["gpu"] == "A100-80"
    assert r["label"].startswith("modeled")  # caveat travels onto the dollar number
