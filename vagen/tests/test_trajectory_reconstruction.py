"""Tests for trajectory reconstruction and integrity checking."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto


def _implementation():
    from vagen.custom_advantage.no_concat_episode_grpo import (
        compute_no_concat_episode_grpo,
    )
    return compute_no_concat_episode_grpo


def _data(rows, width=3):
    """Build a DataProto from row specifications."""
    rewards, masks = [], []
    for row in rows:
        mask = [1] * row["token_count"] + [0] * (width - row["token_count"])
        score = [0.0] * width
        score[row["token_count"] - 1] = row["reward"]
        rewards.append(score)
        masks.append(mask)
    return DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.tensor(rewards, dtype=torch.float32),
                "response_mask": torch.tensor(masks, dtype=torch.float32),
            },
            batch_size=[len(rows)],
        ),
        non_tensor_batch={
            "group_idx": np.array([row["group"] for row in rows], dtype=object),
            "traj_idx": np.array([row["trajectory"] for row in rows]),
            "turn_idx": np.array([row["turn"] for row in rows]),
            "traj_success": np.array([float(row["success"]) for row in rows]),
            "last_turn": np.array([bool(row["last"]) for row in rows]),
        },
        meta_info={},
    )


def _config(**overrides):
    config = {
        "reward_mode": "outcome",
        "loss_weighting": "token",
        "incomplete_group_action": "error",
        "success_reward": 1.0,
        "process_reward_cap": 0.2,
        "format_reward": 0.1,
        "std_epsilon": 1e-6,
    }
    config.update(overrides)
    return {"no_concat_episode_grpo": config}


def test_trajectory_reconstruction_rejects_non_contiguous_turns():
    """Turn indices must be contiguous starting from 1."""
    compute = _implementation()
    rows = [
        {"group": "g", "trajectory": 0, "turn": 1, "reward": 0.0, "success": False, "last": False, "token_count": 2},
        {"group": "g", "trajectory": 0, "turn": 3, "reward": 0.0, "success": False, "last": True, "token_count": 2},
        {"group": "g", "trajectory": 1, "turn": 1, "reward": 1.1, "success": True, "last": True, "token_count": 2},
    ]
    with pytest.raises(ValueError, match="contiguous"):
        compute(
            data=_data(rows),
            gamma=1.0,
            lam=1.0,
            num_repeat=2,
            norm_adv_by_std_in_grpo=True,
            config=_config(),
        )


def test_trajectory_reconstruction_rejects_turns_not_starting_at_one():
    """Trajectories must start at turn 1."""
    compute = _implementation()
    rows = [
        {"group": "g", "trajectory": 0, "turn": 0, "reward": 0.0, "success": False, "last": True, "token_count": 2},
        {"group": "g", "trajectory": 1, "turn": 1, "reward": 1.1, "success": True, "last": True, "token_count": 2},
    ]
    with pytest.raises(ValueError, match="start at 1"):
        compute(
            data=_data(rows),
            gamma=1.0,
            lam=1.0,
            num_repeat=2,
            norm_adv_by_std_in_grpo=True,
            config=_config(),
        )


def test_trajectory_reconstruction_requires_exactly_one_terminal_marker():
    """Each trajectory must have exactly one terminal marker on its last turn."""
    compute = _implementation()
    # No terminal marker
    rows_no_terminal = [
        {"group": "g", "trajectory": 0, "turn": 1, "reward": 0.0, "success": False, "last": False, "token_count": 2},
        {"group": "g", "trajectory": 1, "turn": 1, "reward": 1.1, "success": True, "last": True, "token_count": 2},
    ]
    with pytest.raises(ValueError, match="terminal"):
        compute(
            data=_data(rows_no_terminal),
            gamma=1.0,
            lam=1.0,
            num_repeat=2,
            norm_adv_by_std_in_grpo=True,
            config=_config(),
        )

    # Multiple terminal markers
    rows_multi_terminal = [
        {"group": "g", "trajectory": 0, "turn": 1, "reward": 0.0, "success": False, "last": True, "token_count": 2},
        {"group": "g", "trajectory": 0, "turn": 2, "reward": 0.0, "success": False, "last": True, "token_count": 2},
        {"group": "g", "trajectory": 1, "turn": 1, "reward": 1.1, "success": True, "last": True, "token_count": 2},
    ]
    with pytest.raises(ValueError, match="terminal"):
        compute(
            data=_data(rows_multi_terminal),
            gamma=1.0,
            lam=1.0,
            num_repeat=2,
            norm_adv_by_std_in_grpo=True,
            config=_config(),
        )


def test_trajectory_reconstruction_validates_trajectory_id_range():
    """Trajectory indices must match [0, num_repeat)."""
    compute = _implementation()
    rows = [
        {"group": "g", "trajectory": 0, "turn": 1, "reward": 0.0, "success": False, "last": True, "token_count": 2},
        {"group": "g", "trajectory": 2, "turn": 1, "reward": 1.1, "success": True, "last": True, "token_count": 2},
    ]
    with pytest.raises(ValueError, match="traj_idx must be exactly"):
        compute(
            data=_data(rows),
            gamma=1.0,
            lam=1.0,
            num_repeat=2,
            norm_adv_by_std_in_grpo=True,
            config=_config(),
        )


def test_trajectory_reconstruction_detects_conflicting_duplicates():
    """Padding duplicates must be exact copies across all fields."""
    compute = _implementation()
    rows = [
        {"group": "g", "trajectory": 0, "turn": 1, "reward": 0.0, "success": False, "last": True, "token_count": 2},
        {"group": "g", "trajectory": 1, "turn": 1, "reward": 1.1, "success": True, "last": True, "token_count": 2},
        {"group": "g", "trajectory": 1, "turn": 1, "reward": 1.2, "success": True, "last": True, "token_count": 2},
    ]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        compute(
            data=_data(rows),
            gamma=1.0,
            lam=1.0,
            num_repeat=2,
            norm_adv_by_std_in_grpo=True,
            config=_config(),
        )


def test_trajectory_reconstruction_handles_multi_turn_trajectories():
    """Multi-turn trajectories should be correctly reconstructed and scored."""
    compute = _implementation()
    rows = [
        {"group": "g", "trajectory": 0, "turn": 1, "reward": 0.1, "success": False, "last": False, "token_count": 2},
        {"group": "g", "trajectory": 0, "turn": 2, "reward": 0.1, "success": False, "last": False, "token_count": 2},
        {"group": "g", "trajectory": 0, "turn": 3, "reward": 0.1, "success": False, "last": True, "token_count": 2},
        {"group": "g", "trajectory": 1, "turn": 1, "reward": 1.1, "success": True, "last": True, "token_count": 2},
    ]
    advantages, _ = compute(
        data=_data(rows),
        gamma=1.0,
        lam=1.0,
        num_repeat=2,
        norm_adv_by_std_in_grpo=True,
        config=_config(),
    )
    # Both trajectories should have advantages assigned
    assert torch.any(advantages[:3].ne(0))
    assert torch.any(advantages[3:].ne(0))


def test_trajectory_reconstruction_reports_metadata():
    """Reconstruction should report group, trajectory, and duplicate counts."""
    compute = _implementation()
    rows = [
        {"group": "g1", "trajectory": 0, "turn": 1, "reward": 0.0, "success": False, "last": True, "token_count": 2},
        {"group": "g1", "trajectory": 1, "turn": 1, "reward": 1.1, "success": True, "last": True, "token_count": 2},
        {"group": "g1", "trajectory": 1, "turn": 1, "reward": 1.1, "success": True, "last": True, "token_count": 2},
        {"group": "g2", "trajectory": 0, "turn": 1, "reward": 0.0, "success": False, "last": True, "token_count": 2},
        {"group": "g2", "trajectory": 1, "turn": 1, "reward": 1.1, "success": True, "last": True, "token_count": 2},
    ]
    data = _data(rows)
    compute(
        data=data,
        gamma=1.0,
        lam=1.0,
        num_repeat=2,
        norm_adv_by_std_in_grpo=True,
        config=_config(),
    )
    meta = data.meta_info["no_concat_episode_grpo"]
    assert meta["groups"] == 2
    assert meta["valid_groups"] == 2
    assert meta["trajectories"] == 4
    assert meta["padding_duplicates"] == 1


def test_trajectory_reconstruction_preserves_empty_response_detection():
    """Turns with no response tokens should be rejected."""
    compute = _implementation()
    rows = [
        {"group": "g", "trajectory": 0, "turn": 1, "reward": 0.0, "success": False, "last": True, "token_count": 0},
        {"group": "g", "trajectory": 1, "turn": 1, "reward": 1.1, "success": True, "last": True, "token_count": 2},
    ]
    with pytest.raises(ValueError, match="no response tokens"):
        compute(
            data=_data(rows),
            gamma=1.0,
            lam=1.0,
            num_repeat=2,
            norm_adv_by_std_in_grpo=True,
            config=_config(),
        )
