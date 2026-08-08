"""The evaluation context protocol must match the training context protocol.

A no-concat policy is trained on ``system prompt + current observation``.
Evaluating it with the accumulated dialogue measures a different deployment,
so the protocol has to be derived from the method, recorded in the evidence,
included in resume identity, and enforced before a result is published.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from vagen.analysis.final_evaluation import aggregate_final_tests
from vagen.evaluate.run_eval import _collect_completed_runs, _job_resume_key


ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = ROOT / "scripts" / "run_visual_eval.sh"


def _run_eval_dry(tmp_path: Path, **environment: str) -> str:
    settings = {
        "DRY_RUN": "1",
        "PYTHON_BIN": "python3",
        "DUMP_DIR": str(tmp_path / "dump"),
        "LOG_DIR": str(tmp_path / "logs"),
        **environment,
    }
    completed = subprocess.run(
        ["bash", str(EVAL_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(tmp_path), **settings},
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("base", "true"),
        ("concat_grpo", "true"),
        ("no_concat_gae", "false"),
        ("no_concat_episode_grpo", "false"),
    ],
)
def test_context_protocol_follows_the_training_method(
    tmp_path: Path, method: str, expected: str
) -> None:
    stdout = _run_eval_dry(
        tmp_path / method,
        ENVIRONMENT="sokoban",
        EVAL_METHOD=method,
    )
    assert f"envs.0.concat_multi_turn={expected}" in stdout


def test_unknown_evaluation_method_is_rejected(tmp_path: Path) -> None:
    completed = subprocess.run(
        ["bash", str(EVAL_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(tmp_path),
            "DRY_RUN": "1",
            "PYTHON_BIN": "python3",
            "ENVIRONMENT": "sokoban",
            "EVAL_METHOD": "mystery_method",
            "DUMP_DIR": str(tmp_path / "dump"),
        },
    )
    assert completed.returncode == 1
    assert "EVAL_METHOD must be" in completed.stderr


def test_final_test_cannot_override_the_method_context_protocol(tmp_path: Path) -> None:
    completed = subprocess.run(
        ["bash", str(EVAL_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(tmp_path),
            "DRY_RUN": "1",
            "PYTHON_BIN": "python3",
            "ENVIRONMENT": "sokoban",
            "EVAL_METHOD": "no_concat_episode_grpo",
            "EVALUATION_ROLE": "final_test",
            "CONCAT_MULTI_TURN": "true",
            "DUMP_DIR": str(tmp_path / "dump"),
        },
    )
    assert completed.returncode == 2
    assert "cannot override the context protocol" in completed.stderr


def test_formal_evaluation_configs_declare_the_protocol_explicitly() -> None:
    for relative in (
        "examples/evaluate/sokoban/config.yaml",
        "examples/evaluate/navigation/config_base.yaml",
    ):
        config = yaml.safe_load((ROOT / relative).read_text())
        environment = config["envs"][0]
        assert "concat_multi_turn" in environment, relative


def test_matrix_anti_cheat_requires_an_explicit_method() -> None:
    source = (ROOT / "scripts" / "run_experiment_matrix.sh").read_text()
    anti_cheat = source.split("anti-cheat)", 1)[1].split(";;", 1)[0]
    assert "EVAL_METHOD" in anti_cheat
    assert re.search(r'if \[\[ -z "\$\{EVAL_METHOD:-\}" \]\]', anti_cheat)


def test_resume_identity_separates_the_two_context_protocols(tmp_path: Path) -> None:
    rollout = tmp_path / "tag_sokoban_test" / "rollout_0"
    rollout.mkdir(parents=True)
    (rollout / "metrics.json").write_text(
        json.dumps({"finish_reason": "done", "success": True, "terminated": True})
    )
    (rollout / "meta.json").write_text(
        json.dumps(
            {
                "env_name": "Sokoban",
                "seed": 20003,
                "tag_id": "sokoban_test",
                "observation_ablation": "none",
                "concat_multi_turn": True,
            }
        )
    )

    completed = _collect_completed_runs(str(tmp_path))
    concat_job = {
        "env_name": "Sokoban",
        "seed": 20003,
        "tag_id": "sokoban_test",
        "observation_ablation": "none",
        "concat_multi_turn": True,
    }
    no_concat_job = {**concat_job, "concat_multi_turn": False}

    assert _job_resume_key(concat_job) in completed
    assert _job_resume_key(no_concat_job) not in completed


def test_episode_without_a_recorded_protocol_never_satisfies_resume(tmp_path: Path) -> None:
    rollout = tmp_path / "tag_sokoban_test" / "rollout_0"
    rollout.mkdir(parents=True)
    (rollout / "metrics.json").write_text(
        json.dumps({"finish_reason": "done", "success": True, "terminated": True})
    )
    (rollout / "meta.json").write_text(
        json.dumps(
            {
                "env_name": "Sokoban",
                "seed": 20003,
                "tag_id": "sokoban_test",
                "observation_ablation": "none",
            }
        )
    )
    assert _collect_completed_runs(str(tmp_path)) == {}


def _final_test_fixture(tmp_path: Path, *, training_concat: bool, eval_concat: bool) -> Path:
    training = tmp_path / "training"
    (training / "gpu_metrics").mkdir(parents=True)
    (training / "manifest.json").write_text(
        json.dumps(
            {
                "method": "no_concat_episode_grpo",
                "environment": "sokoban",
                "seed": 0,
                "commit": "abc123",
                "concat_multi_turn": training_concat,
            }
        )
    )
    (training / "parity.json").write_text(
        json.dumps(
            {
                "gate_passed": True,
                "metrics": {
                    "ratio_p95": 0.98,
                    "ratio_p99": 1.02,
                    "mean_abs_logprob_delta": 0.01,
                },
            }
        )
    )

    evaluation = tmp_path / "final_test"
    evaluation.mkdir(parents=True)
    (evaluation / "manifest.json").write_text(
        json.dumps(
            {
                "evaluation_role": "final_test",
                "environment": "sokoban",
                "observation_ablation": "none",
                "concat_multi_turn": eval_concat,
                "model": str(tmp_path / "model"),
                "source_run_dir": str(training),
                "source_selection_manifest": str(tmp_path / "selection.json"),
                "source_export_manifest": str(tmp_path / "export.json"),
                "source_method": "no_concat_episode_grpo",
                "source_environment": "sokoban",
                "source_train_seed": 0,
                "source_checkpoint_step": 400,
            }
        )
    )
    return evaluation


def test_mismatched_context_protocol_cannot_be_published(tmp_path: Path) -> None:
    evaluation = _final_test_fixture(tmp_path, training_concat=False, eval_concat=True)
    result = aggregate_final_tests(
        [evaluation],
        tmp_path / "out",
        expected_methods=("no_concat_episode_grpo",),
        expected_environments=("sokoban",),
    )
    issues = " ".join(str(row.get("Integrity Issues", "")) for row in result["per_run"])
    assert "context protocol does not match training" in issues
    assert result["registry_complete"] is False
    assert all(row["Status"] != "complete" for row in result["per_run"])


def test_final_test_manifest_without_a_protocol_is_rejected(tmp_path: Path) -> None:
    evaluation = _final_test_fixture(tmp_path, training_concat=False, eval_concat=False)
    manifest_path = evaluation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["concat_multi_turn"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="concat_multi_turn"):
        aggregate_final_tests(
            [evaluation],
            tmp_path / "out",
            expected_methods=("no_concat_episode_grpo",),
            expected_environments=("sokoban",),
        )
