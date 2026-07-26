"""Trajectory-aware group-relative advantages for no-concat rollouts.

Each batch row is one environment turn.  Group statistics are computed only
after exact padding duplicates are removed and rows are reconstructed into
complete ``(group_idx, traj_idx)`` trajectories.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


def _value(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _section(config: Any) -> Any:
    return _value(config, "no_concat_episode_grpo", {})


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _group_key(value: Any) -> tuple[str, str]:
    value = _python_scalar(value)
    return type(value).__name__, str(value)


def _as_1d(values: Any, name: str, expected: int) -> np.ndarray:
    result = np.asarray(values).reshape(-1)
    if len(result) != expected:
        raise ValueError(f"{name} has {len(result)} rows; expected {expected}")
    return result


def trajectory_reward_from_turns(
    turn_rewards: torch.Tensor,
    success: bool,
    *,
    mode: str,
    success_reward: float = 1.0,
    process_reward_cap: float = 0.2,
    format_reward: float = 0.1,
    tolerance: float = 1e-6,
) -> float:
    """Reduce turn rewards to one trajectory score with explicit bias control."""
    rewards = torch.as_tensor(turn_rewards, dtype=torch.float64).reshape(-1)
    if rewards.numel() == 0:
        raise ValueError("a trajectory must contain at least one turn reward")

    outcome = float(bool(success))
    if mode == "outcome":
        return outcome

    process_rewards = rewards.clone()
    if success:
        # VAGEN adds the task success bonus to the terminal turn reward.
        process_rewards[-1] -= float(success_reward)

    if mode == "bounded_process":
        process_total = float(process_rewards.sum().item())
        cap = float(process_reward_cap)
        if cap < 0:
            raise ValueError("process_reward_cap must be non-negative")
        return outcome + min(max(process_total, -cap), cap)

    if mode == "format_gate":
        threshold = float(format_reward) - float(tolerance)
        format_is_valid = bool(torch.all(process_rewards >= threshold).item())
        return outcome if format_is_valid else 0.0

    raise ValueError(f"unknown trajectory reward mode: {mode!r}")


def compute_policy_weights(
    response_mask: torch.Tensor,
    group_idx: Any,
    traj_idx: Any,
    turn_idx: Any,
    *,
    mode: str,
    row_is_active: Any | None = None,
) -> torch.Tensor:
    """Return normalized token weights for token/turn/trajectory objectives.

    The returned weights sum to one over active response tokens.  Exact
    ``(group, trajectory, turn)`` duplicates receive zero weight, making the
    objective invariant to DP padding.
    """
    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must be rank 2, got shape {tuple(response_mask.shape)}")
    row_count = response_mask.shape[0]
    groups = _as_1d(group_idx, "group_idx", row_count)
    trajectories = _as_1d(traj_idx, "traj_idx", row_count)
    turns = _as_1d(turn_idx, "turn_idx", row_count)
    active = (
        np.ones(row_count, dtype=bool)
        if row_is_active is None
        else _as_1d(row_is_active, "row_is_active", row_count).astype(bool)
    )

    first_by_turn: dict[tuple[tuple[str, str], int, int], int] = {}
    for row in range(row_count):
        key = (_group_key(groups[row]), int(trajectories[row]), int(turns[row]))
        first_by_turn.setdefault(key, row)

    unique_rows = [row for row in first_by_turn.values() if active[row]]
    weights = torch.zeros_like(response_mask, dtype=torch.float32)
    if not unique_rows:
        return weights

    mask = response_mask.to(dtype=torch.bool)
    token_count = {row: int(mask[row].sum().item()) for row in unique_rows}
    empty_rows = [row for row, count in token_count.items() if count == 0]
    if empty_rows:
        raise ValueError(f"active turns have no response tokens: rows {empty_rows[:5]}")

    if mode == "token":
        normalizer = float(sum(token_count.values()))
        for row in unique_rows:
            weights[row, mask[row]] = 1.0 / normalizer
        return weights

    if mode == "turn":
        normalizer = float(len(unique_rows))
        for row in unique_rows:
            weights[row, mask[row]] = 1.0 / (normalizer * token_count[row])
        return weights

    if mode == "trajectory":
        rows_by_trajectory: dict[tuple[tuple[str, str], int], list[int]] = defaultdict(list)
        for row in unique_rows:
            trajectory_key = (_group_key(groups[row]), int(trajectories[row]))
            rows_by_trajectory[trajectory_key].append(row)
        normalizer = float(len(rows_by_trajectory))
        for rows in rows_by_trajectory.values():
            trajectory_tokens = sum(token_count[row] for row in rows)
            for row in rows:
                weights[row, mask[row]] = 1.0 / (normalizer * trajectory_tokens)
        return weights

    raise ValueError(f"unknown policy weighting mode: {mode!r}")


def _rows_match(
    left: int,
    right: int,
    batch: Any,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    successes: np.ndarray,
    terminal: np.ndarray,
) -> bool:
    core_fields_match = (
        torch.equal(rewards[left], rewards[right])
        and torch.equal(response_mask[left], response_mask[right])
        and bool(successes[left]) == bool(successes[right])
        and bool(terminal[left]) == bool(terminal[right])
    )
    if not core_fields_match:
        return False
    for key in (
        "responses",
        "input_ids",
        "attention_mask",
        "position_ids",
        "rollout_log_probs",
    ):
        if key in batch and not torch.equal(batch[key][left], batch[key][right]):
            return False
    return True


@register_adv_est("no_concat_episode_grpo")
def compute_no_concat_episode_grpo(
    data,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one group-relative advantage per complete trajectory."""
    del gamma, lam
    if int(num_repeat) < 2:
        raise ValueError("no_concat_episode_grpo requires rollout.n >= 2")

    rewards = data.batch.get("token_level_rewards", data.batch.get("token_level_scores"))
    if rewards is None:
        raise KeyError("token_level_rewards or token_level_scores is required")
    response_mask = data.batch["response_mask"]
    if rewards.shape != response_mask.shape:
        raise ValueError(
            f"reward/mask shape mismatch: {tuple(rewards.shape)} vs {tuple(response_mask.shape)}"
        )

    row_count = rewards.shape[0]
    groups = _as_1d(data.non_tensor_batch["group_idx"], "group_idx", row_count)
    trajectories = _as_1d(data.non_tensor_batch["traj_idx"], "traj_idx", row_count)
    turns = _as_1d(data.non_tensor_batch["turn_idx"], "turn_idx", row_count)
    successes = _as_1d(data.non_tensor_batch["traj_success"], "traj_success", row_count)
    terminal = _as_1d(data.non_tensor_batch["last_turn"], "last_turn", row_count)

    first_by_turn: dict[tuple[tuple[str, str], int, int], int] = {}
    duplicate_count = 0
    for row in range(row_count):
        key = (_group_key(groups[row]), int(trajectories[row]), int(turns[row]))
        if key not in first_by_turn:
            first_by_turn[key] = row
            continue
        first = first_by_turn[key]
        if not _rows_match(
            first,
            row,
            data.batch,
            rewards,
            response_mask,
            successes,
            terminal,
        ):
            raise ValueError(f"conflicting duplicate for (group, trajectory, turn)={key}")
        duplicate_count += 1

    unique_rows = list(first_by_turn.values())
    rows_by_trajectory: dict[tuple[tuple[str, str], int], list[int]] = defaultdict(list)
    for row in unique_rows:
        rows_by_trajectory[(_group_key(groups[row]), int(trajectories[row]))].append(row)

    issues_by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
    for trajectory_key, rows in rows_by_trajectory.items():
        rows.sort(key=lambda row: int(turns[row]))
        turn_values = [int(turns[row]) for row in rows]
        expected_turns = list(range(1, len(turn_values) + 1))
        if turn_values != expected_turns:
            issues_by_group[trajectory_key[0]].append(
                f"trajectory {trajectory_key[1]} turn_idx must start at 1 and be contiguous: "
                f"{turn_values}"
            )
        terminal_rows = [row for row in rows if bool(terminal[row])]
        if len(terminal_rows) != 1 or terminal_rows[0] != rows[-1]:
            issues_by_group[trajectory_key[0]].append(
                f"trajectory {trajectory_key[1]} must have one terminal marker on its final turn"
            )

    trajectories_by_group: dict[tuple[str, str], list[tuple[tuple[str, str], int]]] = defaultdict(list)
    for trajectory_key in rows_by_trajectory:
        trajectories_by_group[trajectory_key[0]].append(trajectory_key)
    for group, trajectory_keys in trajectories_by_group.items():
        trajectory_ids = sorted(key[1] for key in trajectory_keys)
        expected_ids = list(range(int(num_repeat)))
        if trajectory_ids != expected_ids:
            issues_by_group[group].append(
                f"group {group[1]!r} traj_idx must be exactly {expected_ids}; "
                f"got {trajectory_ids}"
            )
        if len(trajectory_keys) != int(num_repeat):
            issues_by_group[group].append(
                f"group {group[1]!r} has {len(trajectory_keys)} trajectories; "
                f"expected {int(num_repeat)} trajectories"
            )

    section = _section(config)
    incomplete_action = str(_value(section, "incomplete_group_action", "error"))
    if incomplete_action not in {"error", "drop"}:
        raise ValueError("incomplete_group_action must be 'error' or 'drop'")
    if issues_by_group and incomplete_action == "error":
        first_group = next(iter(issues_by_group))
        raise ValueError("; ".join(issues_by_group[first_group]))
    invalid_groups = set(issues_by_group)

    reward_mode = str(_value(section, "reward_mode", "outcome"))
    success_reward = float(_value(section, "success_reward", 1.0))
    process_reward_cap = float(_value(section, "process_reward_cap", 0.2))
    format_reward = float(_value(section, "format_reward", 0.1))
    std_epsilon = float(_value(section, "std_epsilon", 1e-6))

    turn_rewards = (rewards * response_mask.to(dtype=rewards.dtype)).sum(dim=-1)
    score_by_trajectory: dict[tuple[tuple[str, str], int], float] = {}
    for trajectory_key, rows in rows_by_trajectory.items():
        if trajectory_key[0] in invalid_groups:
            continue
        success = any(bool(successes[row]) for row in rows)
        score_by_trajectory[trajectory_key] = trajectory_reward_from_turns(
            turn_rewards[rows],
            success,
            mode=reward_mode,
            success_reward=success_reward,
            process_reward_cap=process_reward_cap,
            format_reward=format_reward,
        )

    advantage_by_trajectory: dict[tuple[tuple[str, str], int], float] = {}
    zero_variance_groups = 0
    for group, trajectory_keys in trajectories_by_group.items():
        if group in invalid_groups:
            continue
        scores = torch.tensor(
            [score_by_trajectory[key] for key in trajectory_keys],
            dtype=torch.float64,
            device=rewards.device,
        )
        centered = scores - scores.mean()
        if norm_adv_by_std_in_grpo:
            std = scores.std(unbiased=False)
            if float(std.item()) <= std_epsilon:
                centered.zero_()
                zero_variance_groups += 1
            else:
                centered /= std
        for trajectory_key, advantage in zip(trajectory_keys, centered.tolist()):
            advantage_by_trajectory[trajectory_key] = float(advantage)

    advantages = torch.zeros_like(rewards)
    active_rows = np.zeros(row_count, dtype=bool)
    mask_bool = response_mask.to(dtype=torch.bool)
    for trajectory_key, rows in rows_by_trajectory.items():
        if trajectory_key not in advantage_by_trajectory:
            continue
        advantage = advantage_by_trajectory[trajectory_key]
        for row in rows:
            if not bool(mask_bool[row].any()):
                raise ValueError(f"active turn row {row} has no response tokens")
            advantages[row, mask_bool[row]] = advantage
            active_rows[row] = True

    policy_mode = str(_value(section, "loss_weighting", "trajectory"))
    data.batch["policy_weights"] = compute_policy_weights(
        response_mask=response_mask,
        group_idx=groups,
        traj_idx=trajectories,
        turn_idx=turns,
        mode=policy_mode,
        row_is_active=active_rows,
    ).to(device=rewards.device, dtype=rewards.dtype)

    data.meta_info["no_concat_episode_grpo"] = {
        "groups": len(trajectories_by_group),
        "valid_groups": len(trajectories_by_group) - len(invalid_groups),
        "dropped_groups": len(invalid_groups),
        "trajectories": len(score_by_trajectory),
        "padding_duplicates": duplicate_count,
        "zero_variance_groups": zero_variance_groups,
        "reward_mode": reward_mode,
        "loss_weighting": policy_mode,
    }
    return advantages, advantages.clone()
