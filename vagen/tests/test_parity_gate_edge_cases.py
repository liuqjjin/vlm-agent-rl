"""Enhanced tests for parity gate thresholds and error conditions."""

from __future__ import annotations

import math

import pytest
import torch


def _implementation():
    from vagen.utils.logprob_parity import (
        calculate_rollout_train_parity,
        deduplicate_turn_response_mask,
        enforce_rollout_train_parity,
    )
    return calculate_rollout_train_parity, deduplicate_turn_response_mask, enforce_rollout_train_parity


def test_parity_calculation_rejects_mismatched_shapes():
    """All input tensors must have identical shapes."""
    calculate, _, _ = _implementation()

    train = torch.zeros(2, 3)
    rollout = torch.zeros(2, 3)
    mask = torch.ones(2, 3)

    # Mismatched train shape
    with pytest.raises(ValueError, match="identical shapes"):
        calculate(torch.zeros(2, 4), rollout, mask)

    # Mismatched rollout shape
    with pytest.raises(ValueError, match="identical shapes"):
        calculate(train, torch.zeros(3, 3), mask)

    # Mismatched mask shape
    with pytest.raises(ValueError, match="identical shapes"):
        calculate(train, rollout, torch.ones(2, 4))


def test_parity_calculation_rejects_non_finite_values():
    """NaN and inf values should be rejected."""
    calculate, _, _ = _implementation()

    mask = torch.ones(2, 2)

    # NaN in train
    with pytest.raises(ValueError, match="non-finite"):
        calculate(
            torch.tensor([[0.0, float("nan")], [0.0, 0.0]]),
            torch.zeros(2, 2),
            mask,
        )

    # Inf in rollout
    with pytest.raises(ValueError, match="non-finite"):
        calculate(
            torch.zeros(2, 2),
            torch.tensor([[0.0, 0.0], [float("inf"), 0.0]]),
            mask,
        )

    # -inf in train
    with pytest.raises(ValueError, match="non-finite"):
        calculate(
            torch.tensor([[float("-inf"), 0.0], [0.0, 0.0]]),
            torch.zeros(2, 2),
            mask,
        )


def test_parity_calculation_rejects_invalid_clip_bounds():
    """Clip bounds must satisfy 0 < clip_low < 1 < clip_high."""
    calculate, _, _ = _implementation()

    train = torch.zeros(1, 2)
    rollout = torch.zeros(1, 2)
    mask = torch.ones(1, 2)

    # clip_low >= 1
    with pytest.raises(ValueError, match="expected 0 < clip_low < 1 < clip_high"):
        calculate(train, rollout, mask, clip_low=1.0, clip_high=1.2)

    # clip_high <= 1
    with pytest.raises(ValueError, match="expected 0 < clip_low < 1 < clip_high"):
        calculate(train, rollout, mask, clip_low=0.8, clip_high=1.0)

    # clip_low <= 0
    with pytest.raises(ValueError, match="expected 0 < clip_low < 1 < clip_high"):
        calculate(train, rollout, mask, clip_low=0.0, clip_high=1.2)

    # clip_low >= clip_high
    with pytest.raises(ValueError, match="expected 0 < clip_low < 1 < clip_high"):
        calculate(train, rollout, mask, clip_low=0.9, clip_high=0.8)


def test_parity_calculation_handles_extreme_logprob_deltas():
    """Very large logprob deltas should be clamped for ratio calculation."""
    calculate, _, _ = _implementation()

    # Extreme positive delta
    train_high = torch.tensor([[100.0, 0.0]])
    rollout = torch.zeros(1, 2)
    mask = torch.ones(1, 2)

    metrics = calculate(train_high, rollout, mask)
    # Ratio should be clamped to prevent overflow
    assert math.isfinite(metrics["ratio_mean"])
    assert math.isfinite(metrics["ratio_p95"])
    assert math.isfinite(metrics["ratio_p99"])

    # Extreme negative delta
    train_low = torch.tensor([[-100.0, 0.0]])
    metrics = calculate(train_low, rollout, mask)
    assert math.isfinite(metrics["ratio_mean"])
    assert metrics["ratio_mean"] > 0  # Should be positive even after clamping


def test_parity_calculation_computes_correct_quantiles():
    """Quantiles should be computed correctly from the distribution."""
    calculate, _, _ = _implementation()

    # Create a distribution with known quantiles
    # Values: 0.5, 1.0, 1.0, 2.0 (ratios after exp)
    train = torch.tensor([[math.log(0.5), math.log(1.0), math.log(1.0), math.log(2.0)]])
    rollout = torch.zeros(1, 4)
    mask = torch.ones(1, 4)

    metrics = calculate(train, rollout, mask, clip_low=0.1, clip_high=10.0)

    assert metrics["ratio_median"] == pytest.approx(1.0)
    assert metrics["ratio_mean"] == pytest.approx(1.125)  # (0.5 + 1.0 + 1.0 + 2.0) / 4
    # P95 should be between the 95th percentile (close to 2.0)
    assert 1.5 < metrics["ratio_p95"] <= 2.0


def test_parity_calculation_handles_all_identical_values():
    """All identical logprobs should produce ratio 1.0 everywhere."""
    calculate, _, _ = _implementation()

    train = torch.zeros(2, 3)
    rollout = torch.zeros(2, 3)
    mask = torch.ones(2, 3)

    metrics = calculate(train, rollout, mask)

    assert metrics["ratio_mean"] == pytest.approx(1.0)
    assert metrics["ratio_median"] == pytest.approx(1.0)
    assert metrics["ratio_p95"] == pytest.approx(1.0)
    assert metrics["ratio_p99"] == pytest.approx(1.0)
    assert metrics["mean_abs_logprob_delta"] == pytest.approx(0.0)
    assert metrics["pre_update_clip_fraction"] == pytest.approx(0.0)


def test_parity_gate_enforcement_validates_all_thresholds():
    """Gate should check all configured thresholds."""
    _, _, enforce = _implementation()

    # Failing P95 (when configured)
    with pytest.raises(RuntimeError, match="ratio P95 deviation"):
        enforce(
            {"ratio_p95": 1.15, "ratio_p99": 1.1, "mean_abs_logprob_delta": 0.05, "pre_update_clip_fraction": 0.01},
            max_p95_ratio_deviation=0.1,
            max_p99_ratio_deviation=0.2,
            max_mean_abs_logprob_delta=0.1,
            max_clip_fraction=0.05,
        )

    # Failing P99
    with pytest.raises(RuntimeError, match="ratio P99 deviation"):
        enforce(
            {"ratio_p95": 1.05, "ratio_p99": 1.25, "mean_abs_logprob_delta": 0.05, "pre_update_clip_fraction": 0.01},
            max_p99_ratio_deviation=0.2,
            max_mean_abs_logprob_delta=0.1,
            max_clip_fraction=0.05,
        )

    # Failing mean abs delta
    with pytest.raises(RuntimeError, match="mean absolute logprob delta"):
        enforce(
            {"ratio_p95": 1.05, "ratio_p99": 1.1, "mean_abs_logprob_delta": 0.15, "pre_update_clip_fraction": 0.01},
            max_p99_ratio_deviation=0.2,
            max_mean_abs_logprob_delta=0.1,
            max_clip_fraction=0.05,
        )

    # Failing clip fraction
    with pytest.raises(RuntimeError, match="pre-update clip fraction"):
        enforce(
            {"ratio_p95": 1.05, "ratio_p99": 1.1, "mean_abs_logprob_delta": 0.05, "pre_update_clip_fraction": 0.1},
            max_p99_ratio_deviation=0.2,
            max_mean_abs_logprob_delta=0.1,
            max_clip_fraction=0.05,
        )


def test_parity_gate_enforcement_allows_passing_metrics():
    """Metrics within all thresholds should pass without error."""
    _, _, enforce = _implementation()

    # All metrics within thresholds
    enforce(
        {"ratio_p95": 1.05, "ratio_p99": 1.1, "mean_abs_logprob_delta": 0.05, "pre_update_clip_fraction": 0.01},
        max_p95_ratio_deviation=0.1,
        max_p99_ratio_deviation=0.2,
        max_mean_abs_logprob_delta=0.1,
        max_clip_fraction=0.05,
    )  # Should not raise


def test_parity_gate_enforcement_handles_optional_p95_check():
    """P95 check should be skipped when max_p95_ratio_deviation is None."""
    _, _, enforce = _implementation()

    # P95 fails but check is disabled
    enforce(
        {"ratio_p95": 2.0, "ratio_p99": 1.1, "mean_abs_logprob_delta": 0.05, "pre_update_clip_fraction": 0.01},
        max_p95_ratio_deviation=None,
        max_p99_ratio_deviation=0.2,
        max_mean_abs_logprob_delta=0.1,
        max_clip_fraction=0.05,
    )  # Should not raise


def test_deduplication_preserves_first_occurrence():
    """Deduplication should keep only the first occurrence of each (group, traj, turn)."""
    _, deduplicate, _ = _implementation()

    response_mask = torch.ones(4, 2)
    unique_mask = deduplicate(
        response_mask,
        group_idx=["g", "g", "g", "g"],
        traj_idx=[0, 0, 1, 1],
        turn_idx=[1, 1, 1, 1],  # Rows 1 and 3 are duplicates
    )

    # First occurrences should be preserved
    assert torch.all(unique_mask[0] > 0)
    assert torch.all(unique_mask[2] > 0)

    # Duplicates should be masked out
    assert torch.all(unique_mask[1] == 0)
    assert torch.all(unique_mask[3] == 0)


def test_deduplication_handles_multi_turn_trajectories():
    """Each (group, traj, turn) triple should be deduplicated independently."""
    _, deduplicate, _ = _implementation()

    response_mask = torch.ones(5, 2)
    unique_mask = deduplicate(
        response_mask,
        group_idx=["g", "g", "g", "g", "g"],
        traj_idx=[0, 0, 0, 0, 0],
        turn_idx=[1, 1, 2, 2, 3],  # Duplicates in turns 1 and 2, unique turn 3
    )

    # First occurrence of each turn preserved
    assert torch.all(unique_mask[0] > 0)  # turn 1
    assert torch.all(unique_mask[2] > 0)  # turn 2
    assert torch.all(unique_mask[4] > 0)  # turn 3

    # Duplicates masked
    assert torch.all(unique_mask[1] == 0)  # turn 1 duplicate
    assert torch.all(unique_mask[3] == 0)  # turn 2 duplicate


def test_deduplication_rejects_wrong_rank_mask():
    """response_mask must be rank 2."""
    _, deduplicate, _ = _implementation()

    with pytest.raises(ValueError, match="rank 2"):
        deduplicate(
            torch.ones(3),  # Rank 1
            group_idx=["g", "g", "g"],
            traj_idx=[0, 0, 0],
            turn_idx=[1, 2, 3],
        )


def test_deduplication_rejects_mismatched_index_lengths():
    """Index arrays must match response_mask row count."""
    _, deduplicate, _ = _implementation()

    response_mask = torch.ones(3, 2)

    with pytest.raises(ValueError, match="expected 3"):
        deduplicate(
            response_mask,
            group_idx=["g", "g"],  # Too short
            traj_idx=[0, 0, 0],
            turn_idx=[1, 2, 3],
        )


def test_parity_metrics_include_token_count():
    """Metrics should include the number of tokens evaluated."""
    calculate, _, _ = _implementation()

    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    metrics = calculate(
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        mask,
    )

    assert metrics["num_tokens"] == 3  # 2 + 1 = 3 valid tokens


def test_parity_calculation_with_single_token():
    """Parity calculation should work with just one valid token."""
    calculate, _, _ = _implementation()

    train = torch.tensor([[math.log(1.5), 0.0]])
    rollout = torch.zeros(1, 2)
    mask = torch.tensor([[1, 0]])

    metrics = calculate(train, rollout, mask)

    assert metrics["num_tokens"] == 1
    assert metrics["ratio_mean"] == pytest.approx(1.5)
    assert metrics["ratio_median"] == pytest.approx(1.5)
    assert metrics["ratio_p95"] == pytest.approx(1.5)
    assert metrics["ratio_p99"] == pytest.approx(1.5)
