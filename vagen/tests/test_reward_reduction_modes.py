"""Tests for reward reduction modes and bias control."""

from __future__ import annotations

import pytest
import torch


def _implementation():
    from vagen.custom_advantage.no_concat_episode_grpo import trajectory_reward_from_turns
    return trajectory_reward_from_turns


def test_outcome_mode_returns_binary_success_only():
    """Outcome mode should ignore process rewards entirely."""
    reward_fn = _implementation()

    # Success case - should always return 1.0
    assert reward_fn(
        torch.tensor([0.1, 0.1, 1.1]),
        success=True,
        mode="outcome",
    ) == pytest.approx(1.0)

    # Failure case - should always return 0.0
    assert reward_fn(
        torch.tensor([0.1, 0.1, 0.1]),
        success=False,
        mode="outcome",
    ) == pytest.approx(0.0)

    # High process rewards don't affect outcome
    assert reward_fn(
        torch.tensor([10.0, 10.0, 10.0]),
        success=False,
        mode="outcome",
    ) == pytest.approx(0.0)


def test_bounded_process_mode_caps_process_rewards():
    """Bounded process mode should add capped process rewards to outcome."""
    reward_fn = _implementation()

    # Raw total=1.3; removing the terminal success bonus leaves 0.3, capped to 0.2.
    # Final: 1.0 (outcome) + 0.2 (capped process) = 1.2
    assert reward_fn(
        torch.tensor([0.1, 0.1, 1.1]),
        success=True,
        mode="bounded_process",
        success_reward=1.0,
        process_reward_cap=0.2,
    ) == pytest.approx(1.2)  # 1.0 (outcome) + min(0.3, 0.2)

    # Success with process rewards exceeding cap
    assert reward_fn(
        torch.tensor([0.3, 0.3, 1.3]),
        success=True,
        mode="bounded_process",
        success_reward=1.0,
        process_reward_cap=0.2,
    ) == pytest.approx(1.2)  # 1.0 (outcome) + 0.2 (capped)

    # Failure with negative process rewards
    assert reward_fn(
        torch.tensor([-0.1, -0.1]),
        success=False,
        mode="bounded_process",
        success_reward=1.0,
        process_reward_cap=0.2,
    ) == pytest.approx(-0.2)  # 0.0 (outcome) + max(-0.2, -0.2) = -0.2


def test_bounded_process_mode_removes_success_bonus_correctly():
    """Success bonus should only be removed from the terminal turn."""
    reward_fn = _implementation()

    # Multi-turn success: only terminal turn gets success bonus removed
    result = reward_fn(
        torch.tensor([0.1, 0.1, 1.1]),  # Last turn has success bonus
        success=True,
        mode="bounded_process",
        success_reward=1.0,
        process_reward_cap=0.5,
    )
    # Process rewards: 0.1 + 0.1 + (1.1 - 1.0) = 0.3
    assert result == pytest.approx(1.3)  # 1.0 (outcome) + 0.3 (process)


def test_bounded_process_mode_with_zero_cap():
    """Zero cap should reduce to outcome mode."""
    reward_fn = _implementation()

    assert reward_fn(
        torch.tensor([0.5, 0.5, 1.5]),
        success=True,
        mode="bounded_process",
        success_reward=1.0,
        process_reward_cap=0.0,
    ) == pytest.approx(1.0)


def test_bounded_process_mode_rejects_negative_cap():
    """Negative cap is invalid."""
    reward_fn = _implementation()

    with pytest.raises(ValueError, match="process_reward_cap must be non-negative"):
        reward_fn(
            torch.tensor([0.1, 1.1]),
            success=True,
            mode="bounded_process",
            process_reward_cap=-0.1,
        )


def test_format_gate_mode_validates_format_compliance():
    """Format gate should return outcome if all process rewards meet threshold."""
    reward_fn = _implementation()

    # All rewards above threshold - should return outcome
    assert reward_fn(
        torch.tensor([0.1, 0.1, 1.1]),
        success=True,
        mode="format_gate",
        success_reward=1.0,
        format_reward=0.1,
    ) == pytest.approx(1.0)

    # One reward below threshold - should return 0.0
    assert reward_fn(
        torch.tensor([0.05, 0.1, 1.1]),
        success=True,
        mode="format_gate",
        success_reward=1.0,
        format_reward=0.1,
    ) == pytest.approx(0.0)

    # Failure with valid format - still returns 0.0 (no success)
    assert reward_fn(
        torch.tensor([0.1, 0.1, 0.1]),
        success=False,
        mode="format_gate",
        format_reward=0.1,
    ) == pytest.approx(0.0)


def test_format_gate_mode_uses_tolerance_for_floating_point_comparison():
    """Format gate should use tolerance to avoid spurious rejections."""
    reward_fn = _implementation()

    # Exactly at threshold with default tolerance (1e-6)
    assert reward_fn(
        torch.tensor([0.1, 1.1]),
        success=True,
        mode="format_gate",
        success_reward=1.0,
        format_reward=0.1,
        tolerance=1e-6,
    ) == pytest.approx(1.0)

    # Just below threshold by less than tolerance
    assert reward_fn(
        torch.tensor([0.0999999, 1.1]),
        success=True,
        mode="format_gate",
        success_reward=1.0,
        format_reward=0.1,
        tolerance=1e-6,
    ) == pytest.approx(1.0)

    # Below threshold by more than tolerance
    assert reward_fn(
        torch.tensor([0.09, 1.1]),
        success=True,
        mode="format_gate",
        success_reward=1.0,
        format_reward=0.1,
        tolerance=1e-6,
    ) == pytest.approx(0.0)


def test_format_gate_mode_checks_all_turns_including_non_terminal():
    """Format gate should validate all turns, not just terminal."""
    reward_fn = _implementation()

    # First turn has invalid format
    assert reward_fn(
        torch.tensor([0.05, 0.1, 1.1]),
        success=True,
        mode="format_gate",
        format_reward=0.1,
    ) == pytest.approx(0.0)

    # Middle turn has invalid format
    assert reward_fn(
        torch.tensor([0.1, 0.05, 1.1]),
        success=True,
        mode="format_gate",
        format_reward=0.1,
    ) == pytest.approx(0.0)


def test_reward_mode_rejects_empty_trajectory():
    """All modes should reject trajectories with no turns."""
    reward_fn = _implementation()

    with pytest.raises(ValueError, match="at least one turn"):
        reward_fn(
            torch.tensor([]),
            success=True,
            mode="outcome",
        )


def test_reward_mode_rejects_unknown_mode():
    """Unknown modes should be rejected."""
    reward_fn = _implementation()

    with pytest.raises(ValueError, match="unknown trajectory reward mode"):
        reward_fn(
            torch.tensor([0.1, 1.1]),
            success=True,
            mode="invented_mode",
        )


def test_reward_modes_handle_single_turn_trajectories():
    """All modes should work correctly with single-turn trajectories."""
    reward_fn = _implementation()

    # Outcome mode
    assert reward_fn(
        torch.tensor([1.1]),
        success=True,
        mode="outcome",
    ) == pytest.approx(1.0)

    # Bounded process mode
    assert reward_fn(
        torch.tensor([1.1]),
        success=True,
        mode="bounded_process",
        success_reward=1.0,
        process_reward_cap=0.2,
    ) == pytest.approx(1.1)  # 1.0 + (1.1 - 1.0) = 1.1

    # Format gate mode
    assert reward_fn(
        torch.tensor([1.1]),
        success=True,
        mode="format_gate",
        success_reward=1.0,
        format_reward=0.1,
    ) == pytest.approx(1.0)


def test_reward_modes_preserve_numerical_precision():
    """Reward computation should maintain reasonable numerical precision."""
    reward_fn = _implementation()

    # Very small process rewards
    small_rewards = torch.tensor([1e-8, 1e-8, 1.0 + 1e-8])
    result = reward_fn(
        small_rewards,
        success=True,
        mode="bounded_process",
        success_reward=1.0,
        process_reward_cap=0.2,
    )
    assert result == pytest.approx(1.0 + 2e-8, abs=1e-7)

    # Large process rewards that need capping
    large_rewards = torch.tensor([1.0, 1.0, 2.0])
    result = reward_fn(
        large_rewards,
        success=True,
        mode="bounded_process",
        success_reward=1.0,
        process_reward_cap=0.2,
    )
    assert result == pytest.approx(1.2)  # Capped at 1.0 + 0.2


def test_bounded_process_symmetrically_caps_positive_and_negative_rewards():
    """Cap should apply symmetrically to both positive and negative process rewards."""
    reward_fn = _implementation()

    # Large positive process rewards
    positive_result = reward_fn(
        torch.tensor([0.5, 0.5]),
        success=False,
        mode="bounded_process",
        process_reward_cap=0.3,
    )
    assert positive_result == pytest.approx(0.3)

    # Large negative process rewards
    negative_result = reward_fn(
        torch.tensor([-0.5, -0.5]),
        success=False,
        mode="bounded_process",
        process_reward_cap=0.3,
    )
    assert negative_result == pytest.approx(-0.3)
