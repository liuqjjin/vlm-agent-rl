from __future__ import annotations

import pytest

from vagen.analysis.state_relative_preflight import analyze_state_relative_rows
from vagen.envs.sokoban.sokoban_env import Sokoban
from vagen.utils.state_anchor import canonical_state_anchor


def _row(group, trajectory, turn, anchor, action, score, last):
    return {
        "group_idx": group,
        "traj_idx": trajectory,
        "turn_idx": turn,
        "state_anchor": anchor,
        "output": action,
        "score": score,
        "last_turn": last,
        "traj_success": float(score > 0),
    }


def test_canonical_state_anchor_includes_remaining_budget():
    state = {"state_anchor": {"player": [1, 2], "box": [2, 2]}}
    assert canonical_state_anchor(state, 3) == (
        '{"box":[2,2],"player":[1,2]}\n[remaining_turns=3]'
    )
    assert canonical_state_anchor({"obs_str": "<image>"}, 3) is None


@pytest.mark.asyncio
async def test_visual_sokoban_exposes_text_anchor_without_changing_visual_input():
    env = Sokoban(
        {
            "render_mode": "vision",
            "dim_room": (6, 6),
            "num_boxes": 1,
            "max_steps": 20,
            "min_solution_steps": None,
        }
    )
    try:
        observation, _ = await env.reset(seed=7)
        assert observation["obs_str"].count("<image>") == 1
        assert "<image>" in observation["multi_modal_input"]
        assert " P " in observation["state_anchor"] or " S " in observation["state_anchor"]
    finally:
        await env.close()


def test_state_relative_preflight_proceeds_only_with_comparable_signal():
    rows = [
        _row("g", 0, 1, "initial\n[remaining_turns=2]", "left", 0.0, False),
        _row("g", 0, 2, "left-state\n[remaining_turns=1]", "up", 0.0, True),
        _row("g", 1, 1, "initial\n[remaining_turns=2]", "right", 0.0, False),
        _row("g", 1, 2, "right-state\n[remaining_turns=1]", "down", 1.0, True),
    ]
    report = analyze_state_relative_rows(
        rows,
        thresholds={
            "min_rows": 4,
            "max_missing_anchor_fraction": 0.0,
            "min_comparable_row_fraction": 0.4,
            "min_action_diversity_fraction": 0.5,
            "min_mean_return_to_go_variance": 0.1,
        },
    )
    assert report["decision"] == "proceed"
    assert report["metrics"]["comparable_groups"] == 1
    assert report["metrics"]["mean_return_to_go_variance"] == pytest.approx(0.25)


def test_state_relative_preflight_stops_on_collapsed_actions_and_returns():
    rows = [
        _row("g", 0, 1, "same\n[remaining_turns=2]", "left", 0.0, True),
        _row("g", 1, 1, "same\n[remaining_turns=2]", "left", 0.0, True),
    ]
    report = analyze_state_relative_rows(
        rows,
        thresholds={
            "min_rows": 2,
            "max_missing_anchor_fraction": 0.0,
            "min_comparable_row_fraction": 1.0,
            "min_action_diversity_fraction": 0.5,
            "min_mean_return_to_go_variance": 0.1,
        },
    )
    assert report["decision"] == "stop"
    assert set(report["reasons"]) == {"actions_vary", "returns_vary"}
