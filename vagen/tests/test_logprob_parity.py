"""Exact CPU tests for rollout/training-forward log-probability parity."""

from __future__ import annotations

import math

import pytest
import torch


def _implementation():
    from vagen.utils.logprob_parity import (
        calculate_rollout_train_parity,
        enforce_rollout_train_parity,
    )

    return calculate_rollout_train_parity, enforce_rollout_train_parity


def test_parity_metrics_use_only_valid_response_tokens():
    calculate, _ = _implementation()
    rollout = torch.zeros(2, 2)
    train = torch.tensor([[0.0, math.log(2.0)], [math.log(0.5), math.log(4.0)]])
    mask = torch.tensor([[1, 1], [1, 0]])

    metrics = calculate(train, rollout, mask, clip_low=0.8, clip_high=1.2)

    assert metrics["ratio_mean"] == pytest.approx((1.0 + 2.0 + 0.5) / 3)
    assert metrics["ratio_median"] == pytest.approx(1.0)
    assert metrics["ratio_p95"] == pytest.approx(1.9)
    assert metrics["ratio_p99"] == pytest.approx(1.98)
    assert metrics["mean_abs_logprob_delta"] == pytest.approx(2 * math.log(2.0) / 3)
    assert metrics["pre_update_clip_fraction"] == pytest.approx(2 / 3)
    assert metrics["num_tokens"] == 3


def test_parity_gate_accepts_close_policies_and_rejects_clear_mismatch():
    calculate, enforce = _implementation()
    mask = torch.ones(1, 4)
    close = calculate(
        torch.tensor([[0.0, 0.005, -0.005, 0.01]]),
        torch.zeros(1, 4),
        mask,
    )
    enforce(close, max_p99_ratio_deviation=0.05, max_mean_abs_logprob_delta=0.02, max_clip_fraction=0.0)

    mismatched = calculate(
        torch.tensor([[0.0, math.log(2.0), math.log(0.5), math.log(4.0)]]),
        torch.zeros(1, 4),
        mask,
    )
    with pytest.raises(RuntimeError, match="parity"):
        enforce(
            mismatched,
            max_p99_ratio_deviation=0.2,
            max_mean_abs_logprob_delta=0.1,
            max_clip_fraction=0.1,
        )


def test_empty_mask_is_rejected():
    calculate, _ = _implementation()
    with pytest.raises(ValueError, match="no valid response tokens"):
        calculate(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2))
