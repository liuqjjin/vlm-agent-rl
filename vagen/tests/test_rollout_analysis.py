from __future__ import annotations

import json

import pytest

from vagen.analysis.analyze_rollouts import (
    analyze_evaluation_episodes,
    analyze_training_rows,
    build_result_row,
    collect_evaluation_episodes,
)


def _write_episode(root, name, *, success, turns, reward, valid):
    folder = root / "tag_test" / name
    folder.mkdir(parents=True)
    metrics = {
        "rollout_id": name,
        "seed": int(name[-1]),
        "success": success,
        "num_turns": turns,
        "cumulative_reward": reward,
        "finish_reason": "done" if success else "max_turns",
        "env_name": "Sokoban",
        "observation_ablation": "none",
        "infos": [
            {"metrics": {"turn_metrics": {"action_is_valid": value}}}
            for value in valid
        ],
    }
    (folder / "metrics.json").write_text(json.dumps(metrics))
    (folder / "assistant_texts.json").write_text(
        json.dumps(["<answer>right</answer>"] * turns)
    )
    (folder / "transcript.txt").write_text("synthetic")


def test_evaluation_analysis_reports_failure_and_action_quality(tmp_path):
    _write_episode(
        tmp_path,
        "run1",
        success=True,
        turns=2,
        reward=1.2,
        valid=[True, True],
    )
    _write_episode(
        tmp_path,
        "run2",
        success=False,
        turns=4,
        reward=0.4,
        valid=[True, False, False, True],
    )
    episodes = collect_evaluation_episodes(tmp_path)
    report = analyze_evaluation_episodes(episodes)
    assert report["success_rate"] == pytest.approx(0.5)
    assert report["mean_turns_successful"] == pytest.approx(2.0)
    assert report["invalid_action_fraction"] == pytest.approx(2 / 6)
    assert report["answer_template_concentration"] == pytest.approx(1.0)


def test_training_analysis_deduplicates_padding_and_measures_group_variance():
    rows = [
        {
            "group_idx": "g",
            "traj_idx": 0,
            "turn_idx": 1,
            "score": 0.0,
            "output": "left",
            "last_turn": False,
            "traj_success": 0.0,
        },
        {
            "group_idx": "g",
            "traj_idx": 0,
            "turn_idx": 2,
            "score": 0.0,
            "output": "up",
            "last_turn": True,
            "traj_success": 0.0,
        },
        {
            "group_idx": "g",
            "traj_idx": 1,
            "turn_idx": 1,
            "score": 1.0,
            "output": "right",
            "last_turn": True,
            "traj_success": 1.0,
        },
    ]
    report = analyze_training_rows(rows + [rows[-1].copy()])
    assert report["padding_duplicates"] == 1
    assert report["trajectories"] == 2
    assert report["success_rate"] == pytest.approx(0.5)
    assert report["mean_group_reward_variance"] == pytest.approx(0.25)


def test_result_row_keeps_missing_gpu_and_parity_metrics_null(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "method": "base",
                "environment": "sokoban",
                "seed_start": 0,
                "commit": "abc",
            }
        )
    )
    _write_episode(
        tmp_path,
        "run3",
        success=True,
        turns=3,
        reward=1.3,
        valid=[True, True, True],
    )
    row = build_result_row(tmp_path)
    assert row["Visual Success"] == pytest.approx(1.0)
    assert row["Mean Turns"] == pytest.approx(3.0)
    assert row["Peak VRAM"] is None
    assert row["Ratio P95"] is None
