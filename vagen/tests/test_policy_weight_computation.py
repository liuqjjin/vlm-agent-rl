"""Tests for policy weight computation and normalization."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _implementation():
    from vagen.custom_advantage.no_concat_episode_grpo import compute_policy_weights
    return compute_policy_weights


def test_token_mode_weights_sum_to_one_over_all_tokens():
    """Token mode should distribute weight uniformly across all response tokens."""
    weights_fn = _implementation()

    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.float32)
    weights = weights_fn(
        response_mask=response_mask,
        group_idx=np.array(["g", "g", "g"], dtype=object),
        traj_idx=np.array([0, 0, 1]),
        turn_idx=np.array([1, 2, 1]),
        mode="token",
    )

    # Total weight should be 1.0
    assert weights.sum().item() == pytest.approx(1.0)

    # Each token should have equal weight
    total_tokens = 6  # 2 + 3 + 1
    expected_weight = 1.0 / total_tokens
    assert weights[0, 0].item() == pytest.approx(expected_weight)
    assert weights[0, 1].item() == pytest.approx(expected_weight)
    assert weights[1, 0].item() == pytest.approx(expected_weight)


def test_turn_mode_weights_sum_to_one_over_all_turns():
    """Turn mode should give each turn equal weight regardless of token count."""
    weights_fn = _implementation()

    response_mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.float32)
    weights = weights_fn(
        response_mask=response_mask,
        group_idx=np.array(["g", "g"], dtype=object),
        traj_idx=np.array([0, 0]),
        turn_idx=np.array([1, 2]),
        mode="turn",
    )

    # Total weight should be 1.0
    assert weights.sum().item() == pytest.approx(1.0)

    # Each turn should contribute 1/2 of total weight
    turn1_weight = weights[0].sum().item()
    turn2_weight = weights[1].sum().item()
    assert turn1_weight == pytest.approx(0.5)
    assert turn2_weight == pytest.approx(0.5)

    # Within each turn, tokens are equally weighted
    assert weights[0, 0].item() == pytest.approx(0.5)  # Only token in turn 1
    assert weights[1, 0].item() == pytest.approx(0.25)  # Half of turn 2's weight
    assert weights[1, 1].item() == pytest.approx(0.25)


def test_trajectory_mode_weights_sum_to_one_over_all_trajectories():
    """Trajectory mode should give each trajectory equal weight."""
    weights_fn = _implementation()

    # Trajectory 0: 2 tokens total (2 tokens in turn 1)
    # Trajectory 1: 3 tokens total (1 token in turn 1, 2 tokens in turn 2)
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=torch.float32)
    weights = weights_fn(
        response_mask=response_mask,
        group_idx=np.array(["g", "g", "g"], dtype=object),
        traj_idx=np.array([0, 1, 1]),
        turn_idx=np.array([1, 1, 2]),
        mode="trajectory",
    )

    # Total weight should be 1.0
    assert weights.sum().item() == pytest.approx(1.0)

    # Each trajectory should contribute 1/2 of total weight
    traj0_weight = weights[0].sum().item()
    traj1_weight = (weights[1] + weights[2]).sum().item()
    assert traj0_weight == pytest.approx(0.5)
    assert traj1_weight == pytest.approx(0.5)

    # Within trajectory 0 (2 tokens), each token gets 0.5/2 = 0.25
    assert weights[0, 0].item() == pytest.approx(0.25)
    assert weights[0, 1].item() == pytest.approx(0.25)

    # Within trajectory 1 (3 tokens), each token gets 0.5/3 ≈ 0.1667
    assert weights[1, 0].item() == pytest.approx(0.5 / 3)
    assert weights[2, 0].item() == pytest.approx(0.5 / 3)
    assert weights[2, 1].item() == pytest.approx(0.5 / 3)


def test_policy_weights_ignore_padding_duplicates():
    """Padding duplicates should receive zero weight."""
    weights_fn = _implementation()

    # Row 2 is a duplicate of row 1
    response_mask = torch.tensor([[1, 0], [1, 1], [1, 1]], dtype=torch.float32)
    weights = weights_fn(
        response_mask=response_mask,
        group_idx=np.array(["g", "g", "g"], dtype=object),
        traj_idx=np.array([0, 1, 1]),
        turn_idx=np.array([1, 1, 1]),
        mode="token",
    )

    # Duplicate row should have zero weight
    assert weights[2].sum().item() == pytest.approx(0.0)

    # Non-duplicate rows should still sum to 1.0
    assert (weights[0] + weights[1]).sum().item() == pytest.approx(1.0)


def test_policy_weights_handle_row_is_active_mask():
    """row_is_active parameter should exclude inactive rows."""
    weights_fn = _implementation()

    response_mask = torch.tensor([[1, 0], [1, 1], [1, 0]], dtype=torch.float32)
    active = np.array([True, True, False])
    weights = weights_fn(
        response_mask=response_mask,
        group_idx=np.array(["g", "g", "g"], dtype=object),
        traj_idx=np.array([0, 0, 1]),
        turn_idx=np.array([1, 2, 1]),
        mode="token",
        row_is_active=active,
    )

    # Inactive row should have zero weight
    assert weights[2].sum().item() == pytest.approx(0.0)

    # Active rows should sum to 1.0
    assert (weights[0] + weights[1]).sum().item() == pytest.approx(1.0)


def test_policy_weights_reject_empty_turns():
    """Turns with no response tokens should be rejected."""
    weights_fn = _implementation()

    response_mask = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="no response tokens"):
        weights_fn(
            response_mask=response_mask,
            group_idx=np.array(["g", "g"], dtype=object),
            traj_idx=np.array([0, 0]),
            turn_idx=np.array([1, 2]),
            mode="token",
        )


def test_policy_weights_reject_unknown_mode():
    """Unknown weighting modes should be rejected."""
    weights_fn = _implementation()

    response_mask = torch.tensor([[1, 1]], dtype=torch.float32)
    with pytest.raises(ValueError, match="unknown policy weighting mode"):
        weights_fn(
            response_mask=response_mask,
            group_idx=np.array(["g"], dtype=object),
            traj_idx=np.array([0]),
            turn_idx=np.array([1]),
            mode="invented_mode",
        )


def test_policy_weights_require_correct_array_lengths():
    """Index arrays must match response_mask row count."""
    weights_fn = _implementation()

    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="group_idx"):
        weights_fn(
            response_mask=response_mask,
            group_idx=np.array(["g"], dtype=object),  # Too short
            traj_idx=np.array([0, 0]),
            turn_idx=np.array([1, 2]),
            mode="token",
        )


def test_policy_weights_handle_multiple_groups():
    """Each group should be weighted independently."""
    weights_fn = _implementation()

    response_mask = torch.tensor([[1, 1], [1, 0], [1, 1], [1, 0]], dtype=torch.float32)
    weights = weights_fn(
        response_mask=response_mask,
        group_idx=np.array(["g1", "g1", "g2", "g2"], dtype=object),
        traj_idx=np.array([0, 1, 0, 1]),
        turn_idx=np.array([1, 1, 1, 1]),
        mode="token",
    )

    # All tokens should be equally weighted in token mode
    assert weights.sum().item() == pytest.approx(1.0)
    total_tokens = 6  # 2 + 1 + 2 + 1 = 6 actual valid tokens (mask counts)
    expected_weight = 1.0 / total_tokens
    assert weights[0, 0].item() == pytest.approx(expected_weight)


def test_policy_weights_preserve_numerical_stability():
    """Weights should remain stable with many tokens and trajectories."""
    weights_fn = _implementation()

    # Create a large batch
    n_trajectories = 10
    tokens_per_turn = 100
    response_mask = torch.ones(n_trajectories, tokens_per_turn, dtype=torch.float32)

    weights = weights_fn(
        response_mask=response_mask,
        group_idx=np.array(["g"] * n_trajectories, dtype=object),
        traj_idx=np.arange(n_trajectories),
        turn_idx=np.ones(n_trajectories),
        mode="token",
    )

    # Should still sum to exactly 1.0
    assert weights.sum().item() == pytest.approx(1.0, rel=1e-6)

    # Each token should have the same weight
    expected_weight = 1.0 / (n_trajectories * tokens_per_turn)
    assert weights[0, 0].item() == pytest.approx(expected_weight, rel=1e-6)


def test_policy_weights_return_zero_tensor_when_no_active_rows():
    """If all rows are inactive, return zero weights."""
    weights_fn = _implementation()

    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
    weights = weights_fn(
        response_mask=response_mask,
        group_idx=np.array(["g", "g"], dtype=object),
        traj_idx=np.array([0, 0]),
        turn_idx=np.array([1, 2]),
        mode="token",
        row_is_active=np.array([False, False]),
    )

    assert weights.sum().item() == pytest.approx(0.0)
    assert torch.all(weights == 0.0).item()


def test_policy_weights_handle_heterogeneous_group_types():
    """Group indices should work with different types when converted to strings."""
    weights_fn = _implementation()

    response_mask = torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.float32)
    # Mix of string and numeric group indices
    weights = weights_fn(
        response_mask=response_mask,
        group_idx=np.array(["group_a", "group_a", "group_b"], dtype=object),
        traj_idx=np.array([0, 1, 0]),
        turn_idx=np.array([1, 1, 1]),
        mode="trajectory",
    )

    # Should work without errors
    assert weights.sum().item() == pytest.approx(1.0)
