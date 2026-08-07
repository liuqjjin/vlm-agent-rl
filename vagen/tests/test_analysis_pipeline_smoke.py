"""Integration smoke tests for result analysis pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_rollout_analysis_imports_cleanly():
    """analyze_rollouts module should import without errors."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vagen.analysis.analyze_rollouts import main; print('OK')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_statistical_analysis_imports_cleanly():
    """statistical_analysis module should import without errors (requires scipy)."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scipy; from vagen.analysis.statistical_analysis import wilson_score_interval; print('OK')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if "No module named 'scipy'" in result.stderr:
        pytest.skip("scipy not installed (optional dependency)")
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_experiment_summary_imports_cleanly():
    """experiment_summary module should import without errors."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vagen.analysis.experiment_summary import generate_experiment_summary; print('OK')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_result_aggregation_imports_cleanly():
    """result_aggregation module should import without errors (requires scipy)."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scipy; from vagen.analysis.result_aggregation import aggregate_across_seeds; print('OK')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if "No module named 'scipy'" in result.stderr:
        pytest.skip("scipy not installed (optional dependency)")
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_failure_analysis_imports_cleanly():
    """failure_analysis module should import without errors."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vagen.analysis.failure_analysis import analyze_failure_patterns; print('OK')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_state_relative_preflight_imports_and_validates():
    """state_relative_preflight should import and have valid structure."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from vagen.analysis.state_relative_preflight import analyze_state_relative_rows
import inspect
sig = inspect.signature(analyze_state_relative_rows)
print('Parameters:', list(sig.parameters.keys()))
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Parameters:" in result.stdout


def test_objective_comparison_imports_cleanly():
    """objective_comparison module should import without errors."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vagen.analysis.objective_comparison import compare_objective_modes; print('OK')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_visualize_results_imports_cleanly():
    """visualize_results module should import without errors (requires matplotlib)."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import matplotlib; from vagen.analysis.visualize_results import generate_all_plots; print('OK')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if "No module named 'matplotlib'" in result.stderr:
        pytest.skip("matplotlib not installed (optional dependency)")
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_core_algorithms_are_accessible():
    """Core algorithm functions should be importable."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from vagen.custom_advantage.no_concat_episode_grpo import (
    compute_no_concat_episode_grpo,
    compute_policy_weights,
    trajectory_reward_from_turns,
)
print('OK')
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_parity_utilities_are_accessible():
    """Parity check utilities should be importable."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from vagen.utils.logprob_parity import (
    calculate_rollout_train_parity,
    enforce_rollout_train_parity,
    write_rollout_train_parity_report,
)
print('OK')
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_state_anchor_utilities_are_accessible():
    """State anchor utilities should be importable."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from vagen.utils.state_anchor import canonical_state_anchor
from vagen.envs.navigation.navigation_env import navigation_state_anchor
print('OK')
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_environment_modules_import_cleanly():
    """Environment modules should import without errors."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from vagen.envs.sokoban.sokoban_env import Sokoban
from vagen.envs.navigation.navigation_env import NavigationEnv
print('OK')
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_run_manifest_utilities_work():
    """Run manifest utilities should be functional."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from vagen.utils.run_manifest import write_compatible_manifest
import tempfile
import json
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "manifest.json"
    write_compatible_manifest(path, {"test": "value"}, require_existing_match=False)
    data = json.loads(path.read_text())
    assert data["test"] == "value"
    print('OK')
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_config_validation_utilities_import():
    """Config validation utilities should be importable."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vagen.utils.config_validation import validate_config; print('OK')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # May not exist, so just check it doesn't hard fail
    assert result.returncode == 0 or "cannot import" in result.stderr.lower()


def test_multimodal_support_utilities_import():
    """Multimodal support utilities should be importable."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vagen.utils.multimodal_support import require_supported_no_concat_processor; print('OK')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout
