"""Contract tests for trajectory-aware no-concat episode GRPO."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto


def _implementation():
    from vagen.custom_advantage.no_concat_episode_grpo import (
        compute_no_concat_episode_grpo,
        compute_policy_weights,
        trajectory_reward_from_turns,
    )

    return compute_no_concat_episode_grpo, compute_policy_weights, trajectory_reward_from_turns


def _row(group, trajectory, turn, reward, success=False, last=True, token_count=2):
    return {
        "group": group,
        "trajectory": trajectory,
        "turn": turn,
        "reward": reward,
        "success": success,
        "last": last,
        "token_count": token_count,
    }


def _data(rows, width=3):
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


def _advantages(rows, **config):
    compute, _, _ = _implementation()
    return compute(
        data=_data(rows),
        gamma=1.0,
        lam=1.0,
        num_repeat=2,
        norm_adv_by_std_in_grpo=True,
        config=_config(**config),
    )[0]


def test_group_statistics_use_trajectories_not_turn_rows():
    rows = [
        _row("g", 0, 1, 0.0, success=False, last=True),
        _row("g", 1, 1, 0.1, success=False, last=False),
        _row("g", 1, 2, 0.1, success=False, last=False),
        _row("g", 1, 3, 1.1, success=True, last=True),
    ]
    advantages = _advantages(rows)
    valid = _data(rows).batch["response_mask"].bool()
    row_values = [float(advantages[i][valid[i]][0]) for i in range(len(rows))]
    assert row_values == pytest.approx([-1.0, 1.0, 1.0, 1.0])


def test_zero_variance_group_produces_finite_zero_advantages():
    rows = [
        _row("g", 0, 1, 1.1, success=True),
        _row("g", 1, 1, 1.1, success=True),
    ]
    advantages = _advantages(rows)
    assert torch.isfinite(advantages).all()
    assert torch.count_nonzero(advantages).item() == 0


def test_padding_duplicate_does_not_change_unique_advantages():
    rows = [
        _row("g", 0, 1, 0.0, success=False),
        _row("g", 1, 1, 1.1, success=True),
    ]
    baseline = _advantages(rows, loss_weighting="trajectory")
    duplicated = _advantages(rows + [dict(rows[-1])], loss_weighting="trajectory")
    torch.testing.assert_close(duplicated[:2], baseline)
    assert torch.count_nonzero(duplicated[2]).item() == 0


def test_incomplete_group_is_rejected_by_default():
    rows = [_row("g", 0, 1, 0.0, success=False)]
    with pytest.raises(ValueError, match="expected 2 trajectories"):
        _advantages(rows)


def test_incomplete_group_can_be_explicitly_dropped():
    rows = [_row("g", 0, 1, 0.0, success=False)]
    compute, _, _ = _implementation()
    advantages, _ = compute(
        data=_data(rows),
        gamma=1.0,
        lam=1.0,
        num_repeat=2,
        norm_adv_by_std_in_grpo=True,
        config=_config(incomplete_group_action="drop"),
    )
    assert torch.count_nonzero(advantages).item() == 0


def test_missing_terminal_marker_and_turn_gap_are_rejected():
    missing_terminal = [
        _row("g", 0, 1, 0.0, last=False),
        _row("g", 1, 1, 1.1, success=True),
    ]
    with pytest.raises(ValueError, match="terminal"):
        _advantages(missing_terminal)

    turn_gap = [
        _row("g", 0, 1, 0.0, last=False),
        _row("g", 0, 3, 0.0, last=True),
        _row("g", 1, 1, 1.1, success=True),
    ]
    with pytest.raises(ValueError, match="contiguous"):
        _advantages(turn_gap)


def test_conflicting_padding_duplicate_is_rejected():
    rows = [
        _row("g", 0, 1, 0.0, success=False),
        _row("g", 1, 1, 1.1, success=True),
        _row("g", 1, 1, 0.1, success=True),
    ]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _advantages(rows)


def test_reward_modes_make_length_bias_explicit_and_bounded():
    _, _, reward_fn = _implementation()
    short = torch.tensor([1.1])
    long = torch.tensor([0.1, 0.1, 1.1])

    assert reward_fn(short, True, mode="outcome") == pytest.approx(1.0)
    assert reward_fn(long, True, mode="outcome") == pytest.approx(1.0)

    short_bounded = reward_fn(
        short, True, mode="bounded_process", success_reward=1.0, process_reward_cap=0.2
    )
    long_bounded = reward_fn(
        long, True, mode="bounded_process", success_reward=1.0, process_reward_cap=0.2
    )
    assert short_bounded == pytest.approx(1.1)
    assert long_bounded == pytest.approx(1.2)

    assert reward_fn(
        long, True, mode="format_gate", success_reward=1.0, format_reward=0.1
    ) == pytest.approx(1.0)
    assert reward_fn(
        torch.tensor([0.0, 1.1]),
        True,
        mode="format_gate",
        success_reward=1.0,
        format_reward=0.1,
    ) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("token", 4.0),
        ("turn", 13.0 / 3.0),
        ("trajectory", 3.0),
    ],
)
def test_policy_weighting_matches_its_declared_objective(mode, expected):
    _, weights_fn, _ = _implementation()
    response_mask = torch.tensor([[1, 0], [1, 1], [1, 0]], dtype=torch.float32)
    losses = torch.tensor([[1.0, 0.0], [3.0, 3.0], [9.0, 0.0]])
    weights = weights_fn(
        response_mask=response_mask,
        group_idx=np.array(["g", "g", "g"], dtype=object),
        traj_idx=np.array([0, 1, 1]),
        turn_idx=np.array([1, 1, 2]),
        mode=mode,
    )
    objective = (losses * response_mask * weights).sum()
    assert objective.item() == pytest.approx(expected)


def test_policy_weighting_is_padding_duplicate_invariant():
    _, weights_fn, _ = _implementation()
    mask = torch.tensor([[1, 0], [1, 1], [1, 0]], dtype=torch.float32)
    losses = torch.tensor([[1.0, 0.0], [3.0, 3.0], [9.0, 0.0]])
    ids = (
        np.array(["g", "g", "g"], dtype=object),
        np.array([0, 1, 1]),
        np.array([1, 1, 2]),
    )
    weights = weights_fn(mask, *ids, mode="trajectory")
    baseline = (losses * mask * weights).sum()

    duplicate_mask = torch.cat([mask, mask[-1:]], dim=0)
    duplicate_losses = torch.cat([losses, losses[-1:]], dim=0)
    duplicate_ids = (
        np.append(ids[0], ids[0][-1]),
        np.append(ids[1], ids[1][-1]),
        np.append(ids[2], ids[2][-1]),
    )
    duplicate_weights = weights_fn(duplicate_mask, *duplicate_ids, mode="trajectory")
    duplicated = (duplicate_losses * duplicate_mask * duplicate_weights).sum()
    assert duplicated.item() == pytest.approx(baseline.item())
    assert torch.count_nonzero(duplicate_weights[-1]).item() == 0


def test_policy_weighted_objective_is_microbatch_partition_invariant():
    from verl.trainer.ppo.core_algos import apply_policy_loss_weights

    response_mask = torch.tensor([[1, 0], [1, 1], [1, 0]], dtype=torch.float32)
    advantages = torch.tensor([[2.0, 0.0], [-1.0, -1.0], [3.0, 0.0]])
    _, weights_fn, _ = _implementation()
    weights = weights_fn(
        response_mask,
        np.array(["g", "g", "g"], dtype=object),
        np.array([0, 1, 1]),
        np.array([1, 1, 2]),
        mode="trajectory",
    )
    token_losses = torch.tensor([[1.0, 0.0], [3.0, 3.0], [9.0, 0.0]])
    expected = (token_losses * advantages * response_mask * weights).sum()

    accumulated = torch.tensor(0.0)
    partitions = [slice(0, 1), slice(1, 3)]
    for part in partitions:
        local_mask = response_mask[part]
        loss_scale_factor = local_mask.shape[0] / response_mask.shape[0]
        local_advantages = apply_policy_loss_weights(
            advantages[part],
            local_mask,
            weights[part],
            loss_scale_factor,
        )
        local_token_mean = (
            token_losses[part] * local_advantages * local_mask
        ).sum() / local_mask.sum()
        accumulated += local_token_mean * loss_scale_factor

    assert accumulated.item() == pytest.approx(expected.item())
