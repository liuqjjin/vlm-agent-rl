from __future__ import annotations

import json

import pytest

from vagen.analysis.analyze_rollouts import (
    analyze_evaluation_episodes,
    analyze_training_rows,
    build_result_row,
    collect_evaluation_episodes,
)
from vagen.utils.run_manifest import classify_run_for_resume


def _write_episode(root, name, *, success, turns, reward, valid, tag="answer"):
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
        json.dumps([f"<{tag}>right</{tag}>"] * turns)
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
        tag="action",
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


def test_training_analysis_prefers_validation_trajectory_num_turns():
    report = analyze_training_rows(
        [
            {"score": 1.0, "traj_success": True, "num_turns": 4},
            {"score": 1.0, "traj_success": True, "__num_turns__": 6},
        ]
    )
    assert report["mean_turns_successful"] == pytest.approx(5.0)


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
    assert row["Status"] == "incomplete-artifacts"

    manifest_only = tmp_path / "manifest_only"
    manifest_only.mkdir()
    (manifest_only / "manifest.json").write_text(
        json.dumps({"method": "base"})
    )
    assert build_result_row(manifest_only)["Status"] == "incomplete-artifacts"


def test_result_row_requires_expected_episode_count_and_provenance(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "method": "base",
                "environment": "sokoban",
                "seed_start": 3,
                "n_envs": 1,
                "commit": "parent",
                "verl_commit": "submodule",
                "git_dirty": False,
            }
        )
    )
    (tmp_path / "eval_command.sh").write_text("true\n")
    (tmp_path / "resolved_config.txt").write_text("resolved\n")
    gpu_dir = tmp_path / "gpu_metrics"
    gpu_dir.mkdir()
    (gpu_dir / "gpu_summary.json").write_text(
        json.dumps(
            {
                "return_code": 0,
                "sample_count": 2,
                "peak_vram_mib": 100.0,
                "gpu_hours": 0.1,
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

    assert row["Status"] == "complete"
    assert classify_run_for_resume(tmp_path) == "complete"
    assert row["Visual Success"] == pytest.approx(1.0)
    assert row["Peak VRAM"] == pytest.approx(100.0)

    gpu_summary_path = gpu_dir / "gpu_summary.json"
    gpu_summary = json.loads(gpu_summary_path.read_text())
    gpu_summary["sampling_errors"] = ["nvidia-smi timeout"]
    gpu_summary_path.write_text(json.dumps(gpu_summary))
    assert build_result_row(tmp_path)["Status"] == "incomplete-artifacts"
    assert classify_run_for_resume(tmp_path) == "tainted-gpu-metrics"
    gpu_summary["sampling_errors"] = []
    gpu_summary_path.write_text(json.dumps(gpu_summary))

    metrics_path = next(tmp_path.rglob("metrics.json"))
    metrics = json.loads(metrics_path.read_text())
    metrics["finish_reason"] = "model_error"
    metrics_path.write_text(json.dumps(metrics))
    assert build_result_row(tmp_path)["Status"] == "failed"
    metrics["finish_reason"] = "done"
    metrics_path.write_text(json.dumps(metrics))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["git_dirty"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert build_result_row(tmp_path)["Status"] == "incomplete-artifacts"


def test_result_row_preserves_a_failed_parity_attempt_across_resume(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "method": "no_concat_grpo",
                "environment": "sokoban",
                "seed": 1,
                "total_steps": 1,
                "advantage_estimator": "grpo",
                "commit": "parent",
                "verl_commit": "submodule",
                "git_dirty": False,
            }
        )
    )
    (tmp_path / "train_command.sh").write_text("true\n")
    (tmp_path / "resolved_config.yaml").write_text("resolved\n")
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    (rollouts / "1.jsonl").write_text(
        json.dumps(
            {
                "group_idx": "g",
                "traj_idx": 0,
                "turn_idx": 1,
                "score": 1.0,
                "last_turn": True,
                "traj_success": 1.0,
            }
        )
        + "\n"
    )
    gpu_dir = tmp_path / "gpu_metrics"
    gpu_dir.mkdir()
    (gpu_dir / "gpu_summary.json").write_text(
        json.dumps(
            {
                "return_code": 0,
                "sample_count": 1,
                "peak_vram_mib": 100.0,
                "gpu_hours": 0.1,
            }
        )
    )
    (tmp_path / "parity.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "gate_enabled": True,
                "gate_passed": True,
                "metrics": {"ratio_p95": 1.0},
                "attempts": [
                    {"global_step": 1, "gate_passed": False},
                    {"global_step": 1, "gate_passed": True},
                ],
            }
        )
    )

    assert build_result_row(tmp_path)["Status"] == "failed"
    assert classify_run_for_resume(tmp_path) == "failed-parity"
