from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts" / "run_training_method.sh"
VISUAL_EVAL_SCRIPT = ROOT / "scripts" / "run_visual_eval.sh"
MATRIX_SCRIPT = ROOT / "scripts" / "run_experiment_matrix.sh"
EVAL_SCRIPTS = {
    environment: (
        ROOT
        / "examples"
        / "evaluate"
        / environment
        / "sglang"
        / "eval_qwen25_vl_3b.sh"
    )
    for environment in ("frozenlake", "sokoban", "navigation")
}


def _dry_run(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    experiment_dir = tmp_path / overrides.get("METHOD", "no_concat_episode_grpo")
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "ENVIRONMENT": "sokoban",
            "EXPERIMENT_DIR": str(experiment_dir),
            "PYTHON_BIN": sys.executable,
        }
    )
    env.update(overrides)
    return subprocess.run(
        ["bash", str(TRAIN_SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_episode_grpo_entrypoint_is_critic_free_and_auditable(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "critic.enable=False" in result.stdout
    assert "critic.model.path=" not in result.stdout
    assert "critic.optim.lr=" not in result.stdout
    assert "algorithm.no_concat_episode_grpo.reward_mode=outcome" in result.stdout
    assert "algorithm.no_concat_episode_grpo.loss_weighting=trajectory" in result.stdout
    assert "trainer.logger=\\[console\\,wandb\\]" in result.stdout

    experiment_dir = tmp_path / "no_concat_episode_grpo"
    manifest = json.loads((experiment_dir / "manifest.json").read_text())
    assert manifest["logger"] == "wandb"
    assert manifest["wandb_mode"] == "offline"
    assert manifest["seed_scope"] == [
        "python_hash",
        "training_dataloader_order",
    ]
    assert manifest["bitwise_cuda_reproducible"] is False
    assert manifest["critic_enabled"] is False
    assert manifest["parity_gate_enabled"] is True
    assert manifest["parity_thresholds"]["max_p95_ratio_deviation"] == 0.1
    assert manifest["filter_enabled"] is False
    assert manifest["max_response_length"] == 512
    assert manifest["commit"]
    assert manifest["verl_commit"]
    assert (experiment_dir / "train_command.sh").stat().st_mode & 0o111
    replay = (experiment_dir / "train_command.sh").read_text()
    assert "WANDB_MODE=offline" in replay
    assert "PYTHONHASHSEED=0" in replay


@pytest.mark.parametrize(
    ("environment", "threshold"),
    [("frozenlake", "0.02"), ("sokoban", "0.1"), ("navigation", "0.01")],
)
def test_episode_format_gate_uses_the_environment_reward_threshold(
    tmp_path: Path, environment: str, threshold: str
) -> None:
    result = _dry_run(
        tmp_path,
        ENVIRONMENT=environment,
        REWARD_MODE="format_gate",
    )
    assert result.returncode == 0, result.stderr
    assert f"algorithm.no_concat_episode_grpo.format_reward={threshold}" in result.stdout


@pytest.mark.parametrize("method", ["concat_grpo", "no_concat_gae"])
def test_non_episode_entrypoints_do_not_leak_episode_overrides(
    tmp_path: Path, method: str
) -> None:
    result = _dry_run(tmp_path, METHOD=method)
    assert result.returncode == 0, result.stderr
    assert "algorithm.no_concat_episode_grpo." not in result.stdout
    if method == "no_concat_gae":
        assert "critic.enable=True" in result.stdout
        assert "critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct" in result.stdout
        assert "critic.optim.lr=1e-5" in result.stdout
    else:
        assert "critic.enable=False" in result.stdout
        assert "critic.model.path=" not in result.stdout
        assert "critic.optim.lr=" not in result.stdout
    manifest = json.loads((tmp_path / method / "manifest.json").read_text())
    assert manifest["reward_mode"] is None
    assert manifest["loss_weighting"] is None


@pytest.mark.parametrize(
    "method", ["concat_grpo", "no_concat_gae", "no_concat_episode_grpo"]
)
def test_training_entrypoint_composes_current_hydra_schema(
    tmp_path: Path, method: str
) -> None:
    result = _dry_run(tmp_path, METHOD=method)
    assert result.returncode == 0, result.stderr
    command = tmp_path / method / "train_command.sh"
    compose = subprocess.run(
        [str(command), "--cfg", "job", "--resolve"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compose.returncode == 0, compose.stderr
    assert f"adv_estimator: {'grpo' if method == 'concat_grpo' else method}" in compose.stdout
    assert "seed: 0" in compose.stdout


@pytest.mark.parametrize("environment", sorted(EVAL_SCRIPTS))
def test_visual_eval_entrypoint_validates_without_starting_a_server(
    tmp_path: Path, environment: str
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "CHECK_CONFIG_ONLY": "1",
            "DUMP_DIR": str(tmp_path / environment / "dump"),
            "LOG_DIR": str(tmp_path / environment / "logs"),
            "PYTHON_BIN": sys.executable,
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(EVAL_SCRIPTS[environment]),
            "envs.0.n_envs=1",
            "run.max_concurrent_jobs=1",
            "run.resume=force_rerun",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Prepared 1 jobs from 1 environment specs." in result.stdout
    assert "Config check passed" in result.stdout


@pytest.mark.parametrize("environment", sorted(EVAL_SCRIPTS))
def test_visual_eval_wrapper_dry_run_declares_auditable_runtime(
    tmp_path: Path, environment: str
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "ENVIRONMENT": environment,
            "DUMP_DIR": str(tmp_path / environment),
            "PYTHON_BIN": sys.executable,
        }
    )
    result = subprocess.run(
        ["bash", str(VISUAL_EVAL_SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "CONFIG_CHECK_OUTPUT=" in result.stdout
    assert "run.resume=force_rerun" in result.stdout
    assert "eval_qwen25_vl_3b.sh" in result.stdout
    expected_count = 30 if environment == "navigation" else 128
    assert f"envs.0.n_envs={expected_count}" in result.stdout
    seed_start = 30 if environment == "navigation" else 10001
    seed_max = seed_start + expected_count - 1
    assert f"envs.0.seed=\\[{seed_start}\\,{seed_max}\\,1\\]" in result.stdout


def test_episode_screening_uses_nine_distinct_run_names(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "PROJECT_NAME": f"matrix_contract_{tmp_path.name}",
            "PYTHON_BIN": sys.executable,
            "SCREENING_STEPS": "1",
            "SCREENING_BATCH_SIZE": "2",
        }
    )
    result = subprocess.run(
        ["bash", str(MATRIX_SCRIPT), "episode-screening"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    names = re.findall(r"trainer\.experiment_name=([^ ]+)", result.stdout)
    assert len(names) == 9
    assert len(set(names)) == 9


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("REWARD_MODE", "invented", "REWARD_MODE must be"),
        ("LOSS_WEIGHTING", "invented", "LOSS_WEIGHTING must be"),
        ("TRAINER_LOGGER", "invented", "TRAINER_LOGGER must be"),
        ("N_GPUS", "2", "restricted to N_GPUS=1"),
        ("ROLLOUT_N", "not-a-number", "ROLLOUT_N must be a positive integer"),
        ("SEED", "-1", "SEED must be a non-negative integer"),
    ],
)
def test_entrypoint_rejects_invalid_control_values(
    tmp_path: Path, variable: str, value: str, message: str
) -> None:
    result = _dry_run(tmp_path, **{variable: value})
    assert result.returncode == 1
    assert message in result.stderr


def test_entrypoint_blocks_qwen3_vl_case_insensitively(tmp_path: Path) -> None:
    result = _dry_run(
        tmp_path,
        MODEL_PATH="qwen/qwen3-vl-4b-instruct",
    )
    assert result.returncode == 2
    assert "Qwen3-VL is blocked" in result.stderr
