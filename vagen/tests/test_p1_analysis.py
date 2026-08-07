"""Basic unit tests for new P1 analysis modules.

Run with: python -m pytest vagen/tests/test_p1_analysis.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


class TestResultAggregation:
    """Tests for result_aggregation module."""

    def test_aggregate_across_seeds_basic(self, tmp_path):
        """Test basic seed aggregation."""
        from vagen.analysis.result_aggregation import aggregate_across_seeds

        # Create mock run directories
        run_dirs = []
        for seed in range(3):
            run_dir = tmp_path / f"run_seed{seed}"
            run_dir.mkdir()

            # Create manifest
            manifest = {
                "method": "concat_grpo",
                "environment": "sokoban",
                "seed": seed,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            # Create validation JSONL
            validation_dir = run_dir / "validation"
            validation_dir.mkdir()
            trajectories = [
                {"traj_success": True, "num_turns": 5 + seed},
                {"traj_success": False, "num_turns": 10},
            ]
            jsonl_path = validation_dir / "100.jsonl"
            with jsonl_path.open("w") as f:
                for traj in trajectories:
                    f.write(json.dumps(traj) + "\n")

            run_dirs.append(run_dir)

        # Test aggregation
        result = aggregate_across_seeds(run_dirs)

        assert result["method"] == "concat_grpo"
        assert result["n_seeds"] == 3
        assert result["success_rate"]["pooled"] == 0.5  # 3/6 trajectories successful
        assert result["success_rate"]["total_successes"] == 3
        assert result["success_rate"]["total_trials"] == 6

    def test_compute_seed_stability_metrics(self):
        """Test stability metrics computation."""
        from vagen.analysis.result_aggregation import compute_seed_stability_metrics

        # Stable case
        success_rates = [0.48, 0.50, 0.52]
        mean_turns = [6.4, 6.5, 6.6]
        stability = compute_seed_stability_metrics(success_rates, mean_turns)

        assert stability["success_rate_cv"] < 0.1  # Low variance
        assert stability["interpretation"] == "highly_stable"

        # High variance case
        success_rates = [0.3, 0.7, 0.5]
        mean_turns = [5.0, 10.0, 7.5]
        stability = compute_seed_stability_metrics(success_rates, mean_turns)

        assert stability["success_rate_cv"] > 0.2
        assert stability["interpretation"] in ["moderate_variance", "high_variance"]


class TestObjectiveComparison:
    """Tests for objective_comparison module."""

    def test_extract_objective_mode(self):
        """Test objective mode extraction from manifest."""
        from vagen.analysis.objective_comparison import extract_objective_mode

        # Explicit mode
        manifest = {"policy_objective": {"aggregation": "trajectory"}}
        assert extract_objective_mode(manifest) == "trajectory"

        # From method name
        manifest = {"method": "no_concat_episode_grpo_token"}
        assert extract_objective_mode(manifest) == "token"

        manifest = {"method": "turn_level_policy"}
        assert extract_objective_mode(manifest) == "turn"

        # Unknown
        manifest = {"method": "unknown_method"}
        assert extract_objective_mode(manifest) == "unknown"


class TestFailureAnalysis:
    """Tests for failure_analysis module."""

    def test_classify_failure_reason(self):
        """Test failure reason classification."""
        from vagen.analysis.failure_analysis import classify_failure_reason

        # Infrastructure error
        episode = {"finish_reason": "model_error", "success": False}
        assert classify_failure_reason(episode) == "infrastructure_error"

        # Timeout with progress
        episode = {
            "finish_reason": "max_turns",
            "success": False,
            "num_turns": 20,
            "cumulative_reward": 0.6,
        }
        assert classify_failure_reason(episode) == "timeout_with_high_progress"

        # No progress
        episode = {
            "finish_reason": "max_turns",
            "success": False,
            "num_turns": 20,
            "cumulative_reward": 0.05,
        }
        assert classify_failure_reason(episode) == "timeout_with_no_progress"

        # Invalid actions
        episode = {
            "finish_reason": "done",
            "success": False,
            "valid_action_count": 0,
            "action_check_count": 5,
        }
        assert classify_failure_reason(episode) == "all_actions_invalid"

    def test_analyze_failure_patterns(self):
        """Test failure pattern analysis."""
        from vagen.analysis.failure_analysis import analyze_failure_patterns

        episodes = [
            {"success": True, "finish_reason": "done"},
            {"success": False, "finish_reason": "max_turns", "num_turns": 20, "cumulative_reward": 0.1},
            {"success": False, "finish_reason": "model_error"},
            {"success": False, "finish_reason": "max_turns", "num_turns": 20, "cumulative_reward": 0.7},
        ]

        patterns = analyze_failure_patterns(episodes)

        assert patterns["total_episodes"] == 4
        assert patterns["total_failures"] == 3
        assert patterns["failure_rate"] == 0.75
        assert patterns["infrastructure_error_rate"] == 0.25
        assert "timeout_with_partial_progress" in patterns["failure_categories"]

    def test_identify_action_repetition_failures(self):
        """Test action repetition detection."""
        from vagen.analysis.failure_analysis import identify_action_repetition_failures

        episodes = [
            {
                "rollout_id": "ep0",
                "assistant_texts": ["left", "left", "left", "left", "left"],
                "success": False,
            },
            {
                "rollout_id": "ep1",
                "assistant_texts": ["up", "down", "left", "right"],
                "success": True,
            },
        ]

        loops = identify_action_repetition_failures(episodes, repetition_threshold=5)

        assert loops["total_episodes"] == 2
        assert loops["episodes_with_loops"] == 1
        assert loops["loop_rate"] == 0.5


class TestExperimentSummary:
    """Tests for experiment_summary module."""

    @patch("vagen.analysis.experiment_summary.aggregate_experiment_matrix")
    @patch("vagen.analysis.experiment_summary.build_result_row")
    def test_generate_experiment_summary(self, mock_build_row, mock_aggregate, tmp_path):
        """Test experiment summary generation."""
        from vagen.analysis.experiment_summary import generate_experiment_summary

        # Create mock experiment directory
        exp_dir = tmp_path / "experiments"
        exp_dir.mkdir()

        run_dir = exp_dir / "run1"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(json.dumps({"method": "concat_grpo"}))

        # Mock aggregation
        mock_aggregate.return_value = {
            "n_runs": 1,
            "n_groups": 1,
            "groups": {
                "concat_grpo": {
                    "success_rate": {"pooled": 0.5, "std": 0.1},
                    "mean_turns": {"pooled_all_episodes": 6.5},
                    "gpu_hours": {"total": 10.0},
                    "n_seeds": 1,
                }
            },
        }

        mock_build_row.return_value = {
            "Method": "concat_grpo",
            "Status": "complete",
            "GPU·h": 10.0,
        }

        # Generate summary
        summary = generate_experiment_summary(exp_dir, include_plots=False)

        assert summary["overview"]["total_runs"] == 1
        assert summary["best_method"]["name"] == "concat_grpo"
        assert summary["best_method"]["success_rate"] == 0.5


class TestConfigValidation:
    """Tests for config_validation module."""

    def test_validate_training_config_valid(self):
        """Test validation of valid training config."""
        from vagen.utils.config_validation import validate_training_config

        config = {
            "trainer": {
                "n_gpus_per_node": 4,
                "nnodes": 1,
                "total_training_steps": 100,
            },
            "actor_rollout_ref": {
                "actor": {"strategy": "fsdp"},
                "rollout": {"n": 4},
            },
            "algorithm": {
                "adv_estimator": "grpo",
            },
            "critic": {"enable": False, "strategy": "fsdp"},
        }

        errors = validate_training_config(config)
        assert len(errors) == 0

    def test_validate_training_config_invalid(self):
        """Test validation catches errors."""
        from vagen.utils.config_validation import validate_training_config

        config = {
            "trainer": {
                "n_gpus_per_node": -1,  # Invalid
                "nnodes": 0,  # Invalid
            },
            "actor_rollout_ref": {
                "actor": {"strategy": "unknown_strategy"},  # Invalid
                "rollout": {"n": 1},
            },
            "algorithm": {
                "adv_estimator": "no_concat_gae",  # Needs critic
            },
            # Missing critic
        }

        errors = validate_training_config(config)
        assert len(errors) > 0
        assert any("n_gpus_per_node" in e for e in errors)
        assert any("nnodes" in e for e in errors)
        assert any("strategy" in e for e in errors)
        assert any("critic" in e for e in errors)

    def test_validate_evaluation_config(self):
        """Test evaluation config validation."""
        from vagen.utils.config_validation import validate_evaluation_config

        # Valid config
        config = {
            "n_envs": 128,
            "environment": "sokoban",
            "model_path": "/path/to/model",
            "backend": "sglang",
        }
        errors = validate_evaluation_config(config)
        assert len(errors) == 0

        # Invalid config
        config = {
            "n_envs": -1,  # Invalid
            "backend": "unknown_backend",  # Invalid
            # Missing required keys
        }
        errors = validate_evaluation_config(config)
        assert len(errors) > 0

    def test_validate_experiment_matrix_config(self):
        """Test experiment matrix validation."""
        from vagen.utils.config_validation import validate_experiment_matrix_config

        import yaml

        config = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "experiments" / "matrix.yaml").read_text()
        )
        errors = validate_experiment_matrix_config(config)
        assert len(errors) == 0

        # Invalid config
        config = {
            "methods": {"unknown_method": {}},
            "environments": {"unknown_env": {}},
            "funnel": {"screen": {"methods": ["missing"], "seeds": ["bad"]}},
        }
        errors = validate_experiment_matrix_config(config)
        assert len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
