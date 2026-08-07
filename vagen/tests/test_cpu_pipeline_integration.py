"""Integration smoke tests for CPU experiment pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CPU_SCRIPT = ROOT / "vagen" / "analysis" / "run_cpu_experiments.py"


def test_cpu_experiment_pipeline_produces_complete_artifacts(tmp_path: Path):
    """End-to-end CPU pipeline should generate all expected outputs."""
    output_dir = tmp_path / "cpu_results"

    result = subprocess.run(
        [
            sys.executable,
            str(CPU_SCRIPT),
            "--output-dir",
            str(output_dir),
            "--seed-start",
            "0",
            "--seed-count",
            "3",  # Small count for fast test
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    # Check main artifacts exist
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "cpu_diagnostics.svg").exists()
    assert (output_dir / "failure_cases.jsonl").exists()

    # Check raw data
    assert (output_dir / "raw" / "value_mask_steps.csv").exists()
    assert (output_dir / "raw" / "sokoban_reward_trajectories.jsonl").exists()
    assert (output_dir / "raw" / "sokoban_reward_pairs.csv").exists()
    assert (output_dir / "raw" / "policy_weight_mass.csv").exists()


def test_cpu_experiment_summary_contains_required_sections(tmp_path: Path):
    """Summary should include metadata, value_mask, reward_bias, and policy sections."""
    output_dir = tmp_path / "cpu_results"

    subprocess.run(
        [
            sys.executable,
            str(CPU_SCRIPT),
            "--output-dir",
            str(output_dir),
            "--seed-count",
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text())

    # Check required sections
    assert "metadata" in summary
    assert "value_mask" in summary
    assert "sokoban_reward_bias" in summary
    assert "policy_weighting" in summary
    assert "scope" in summary

    # Check metadata fields
    assert "git_commit" in summary["metadata"]
    assert "seed" in summary["metadata"]
    assert "torch" in summary["metadata"]

    # Check value_mask results
    assert "masked_ignored_final" in summary["value_mask"]
    assert "legacy_ignored_final" in summary["value_mask"]

    # Check reward bias results
    assert "all_solved" in summary["sokoban_reward_bias"]
    assert "mean_environment_reward_delta" in summary["sokoban_reward_bias"]

    # Check policy weighting results
    assert "token" in summary["policy_weighting"]
    assert "turn" in summary["policy_weighting"]
    assert "trajectory" in summary["policy_weighting"]


def test_cpu_experiment_value_mask_convergence(tmp_path: Path):
    """Value mask experiment should show divergence between masked and legacy."""
    output_dir = tmp_path / "cpu_results"

    subprocess.run(
        [sys.executable, str(CPU_SCRIPT), "--output-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    value_mask = summary["value_mask"]

    # Masked version should preserve ignored position near initial value
    assert abs(value_mask["masked_ignored_final"] - value_mask["masked_ignored_initial"]) < 5.0

    # Legacy version should move ignored position toward supervised target
    # (which is at 2.0, while ignored should stay near 0.5)
    assert abs(value_mask["legacy_ignored_final"] - value_mask["masked_ignored_initial"]) > 10.0


def test_cpu_experiment_sokoban_reward_bias_detection(tmp_path: Path):
    """Sokoban experiment should detect turn-splitting bias."""
    output_dir = tmp_path / "cpu_results"

    subprocess.run(
        [
            sys.executable,
            str(CPU_SCRIPT),
            "--output-dir",
            str(output_dir),
            "--seed-count",
            "5",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    reward_bias = summary["sokoban_reward_bias"]

    # All seeds should solve successfully
    assert reward_bias["all_solved"] is True

    # Mean reward delta should be positive (split gets more reward)
    assert reward_bias["mean_environment_reward_delta"] > 0

    # Most seeds should show positive bias
    assert reward_bias["positive_delta_fraction"] > 0.5


def test_cpu_experiment_policy_weighting_normalization(tmp_path: Path):
    """Policy weighting modes should each sum to 1.0."""
    output_dir = tmp_path / "cpu_results"

    subprocess.run(
        [sys.executable, str(CPU_SCRIPT), "--output-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    policy = summary["policy_weighting"]

    for mode in ["token", "turn", "trajectory"]:
        total_weight = sum(policy[mode].values())
        assert total_weight == pytest.approx(1.0, abs=1e-6)


def test_cpu_experiment_failure_cases_documented(tmp_path: Path):
    """Failure cases should be documented with evidence."""
    output_dir = tmp_path / "cpu_results"

    subprocess.run(
        [sys.executable, str(CPU_SCRIPT), "--output-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    failure_cases = []
    with (output_dir / "failure_cases.jsonl").open() as f:
        for line in f:
            failure_cases.append(json.loads(line))

    # Should have documented failure cases
    assert len(failure_cases) >= 2

    # Each failure should have required fields
    for case in failure_cases:
        assert "id" in case
        assert "kind" in case
        assert "evidence" in case
        assert "consequence" in case


def test_cpu_experiment_raw_data_is_valid(tmp_path: Path):
    """Raw data files should be parseable and non-empty."""
    output_dir = tmp_path / "cpu_results"

    subprocess.run(
        [
            sys.executable,
            str(CPU_SCRIPT),
            "--output-dir",
            str(output_dir),
            "--seed-count",
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    # Check CSV files are valid
    import csv

    with (output_dir / "raw" / "value_mask_steps.csv").open() as f:
        rows = list(csv.DictReader(f))
        assert len(rows) > 0
        assert "variant" in rows[0]
        assert "step" in rows[0]

    # Check JSONL files are valid
    with (output_dir / "raw" / "sokoban_reward_trajectories.jsonl").open() as f:
        for line in f:
            record = json.loads(line)
            assert "seed" in record
            assert "success" in record
            assert "trajectory_scores" in record


def test_cpu_experiment_svg_output_is_valid_xml(tmp_path: Path):
    """SVG output should be valid XML."""
    output_dir = tmp_path / "cpu_results"

    subprocess.run(
        [sys.executable, str(CPU_SCRIPT), "--output-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    svg_content = (output_dir / "cpu_diagnostics.svg").read_text()

    # Basic validation
    assert svg_content.startswith("<svg")
    assert "</svg>" in svg_content
    assert "viewBox" in svg_content

    # Should contain expected elements
    assert "polyline" in svg_content
    assert "circle" in svg_content
    assert "text" in svg_content


def test_cpu_experiment_respects_seed_parameters(tmp_path: Path):
    """Script should respect --seed-start and --seed-count parameters."""
    output_dir = tmp_path / "cpu_results"

    subprocess.run(
        [
            sys.executable,
            str(CPU_SCRIPT),
            "--output-dir",
            str(output_dir),
            "--seed-start",
            "100",
            "--seed-count",
            "3",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text())

    assert summary["sokoban_reward_bias"]["seed_start"] == 100
    assert summary["sokoban_reward_bias"]["seed_count"] == 3

    # Check raw data has correct seeds
    with (output_dir / "raw" / "sokoban_reward_trajectories.jsonl").open() as f:
        seeds = set()
        for line in f:
            seeds.add(json.loads(line)["seed"])

    # Should have seeds 100, 101, 102 (each appears twice: packed and split)
    assert seeds == {100, 101, 102}


def test_cpu_experiment_is_deterministic(tmp_path: Path):
    """Running the same experiment twice should produce identical results."""
    output_dir1 = tmp_path / "run1"
    output_dir2 = tmp_path / "run2"

    for output_dir in [output_dir1, output_dir2]:
        subprocess.run(
            [
                sys.executable,
                str(CPU_SCRIPT),
                "--output-dir",
                str(output_dir),
                "--seed-count",
                "2",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )

    summary1 = json.loads((output_dir1 / "summary.json").read_text())
    summary2 = json.loads((output_dir2 / "summary.json").read_text())

    # Remove timestamp field
    summary1["metadata"].pop("generated_at_utc", None)
    summary2["metadata"].pop("generated_at_utc", None)

    # Core results should be identical
    assert summary1["value_mask"] == summary2["value_mask"]
    assert summary1["sokoban_reward_bias"] == summary2["sokoban_reward_bias"]
    assert summary1["policy_weighting"] == summary2["policy_weighting"]


def test_cpu_experiment_handles_output_dir_creation(tmp_path: Path):
    """Script should create output directory and subdirectories."""
    output_dir = tmp_path / "nested" / "deep" / "path"

    subprocess.run(
        [
            sys.executable,
            str(CPU_SCRIPT),
            "--output-dir",
            str(output_dir),
            "--seed-count",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    assert output_dir.exists()
    assert (output_dir / "raw").exists()
    assert (output_dir / "summary.json").exists()
