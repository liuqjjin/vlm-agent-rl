from __future__ import annotations

import math

import pytest

from vagen.envs.sokoban.patch_sokoban_env import get_shortest_action_path
from vagen.envs.sokoban.sokoban_env import Sokoban


ACTION_NAMES = {1: "up", 2: "down", 3: "left", 4: "right"}


def _response(actions):
    joined = ",".join(ACTION_NAMES[action] for action in actions)
    return f"<think>follow the verified path</think><answer>{joined}</answer>"


async def _run_solution(seed: int, actions_per_turn: int):
    env = Sokoban(
        {
            "render_mode": "text",
            "dim_room": (6, 6),
            "num_boxes": 1,
            "max_steps": 100,
            "max_actions_per_step": 3,
            "prompt_format": "free_think",
            "min_solution_steps": (3, 5),
        }
    )
    try:
        await env.reset(seed=seed)
        solution = get_shortest_action_path(
            env.env.room_fixed,
            env.env.room_state,
            MAX_DEPTH=20,
        )
        assert 3 <= len(solution) <= 5
        total_reward = 0.0
        turns = 0
        success = False
        for start in range(0, len(solution), actions_per_turn):
            _, reward, done, info = await env.step(
                _response(solution[start : start + actions_per_turn])
            )
            turns += 1
            total_reward += reward
            success = bool(info["success"])
            if done:
                break
        return total_reward, turns, success, len(solution)
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_successful_sokoban_reward_increases_when_same_solution_is_split_into_more_turns():
    packed_reward, packed_turns, packed_success, solution_length = await _run_solution(
        seed=19,
        actions_per_turn=3,
    )
    split_reward, split_turns, split_success, repeated_solution_length = await _run_solution(
        seed=19,
        actions_per_turn=1,
    )

    assert packed_success and split_success
    assert repeated_solution_length == solution_length
    assert packed_turns == math.ceil(solution_length / 3)
    assert split_turns == solution_length
    assert packed_reward == pytest.approx(1.0 + 0.1 * packed_turns)
    assert split_reward == pytest.approx(1.0 + 0.1 * split_turns)
    assert split_reward > packed_reward
