from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vagen.analysis.experiment_contract import validate_experiment_contract
from vagen.analysis.final_evaluation import (
    aggregate_final_tests,
    export_checkpoint,
    plan_checkpoint_export,
    publish_final_results,
    select_validation_checkpoint,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _actor_checkpoint(
    run: Path, step: int, *, adapter_state: str = "complete"
) -> Path:
    actor = run / "checkpoints" / f"global_step_{step}" / "actor"
    _write_json(actor / "fsdp_config.json", {"world_size": 1})
    _write_json(actor / "huggingface" / "config.json", {"architectures": ["Test"]})
    (actor / "model_world_size_1_rank_0.pt").write_bytes(b"synthetic")
    if adapter_state in {"complete", "incomplete"}:
        _write_json(
            actor / "lora_adapter" / "adapter_config.json",
            {"r": 4, "lora_alpha": 4},
        )
    if adapter_state == "complete":
        (actor / "lora_adapter" / "adapter_model.safetensors").write_bytes(
            b"synthetic"
        )
    return actor


def test_validation_selection_and_fsdp_lora_export_dry_run(tmp_path: Path) -> None:
    run = tmp_path / "train"
    _write_json(
        run / "manifest.json",
        {
            "method": "no_concat_episode_grpo",
            "environment": "sokoban",
            "seed": 0,
            "advantage_estimator": "no_concat_episode_grpo",
            "validation_n_envs": 2,
            "lora_rank": 4,
        },
    )
    for step, rows in (
        (100, [{"traj_success": True, "num_turns": 4}, {"traj_success": False, "num_turns": 5}]),
        (200, [{"traj_success": True, "num_turns": 3}, {"traj_success": True, "num_turns": 4}]),
    ):
        validation = run / "validation" / f"{step}.jsonl"
        validation.parent.mkdir(parents=True, exist_ok=True)
        validation.write_text("".join(json.dumps(row) + "\n" for row in rows))
        _actor_checkpoint(run, step)

    selection_path = run / "selection" / "checkpoint_selection.json"
    selection = select_validation_checkpoint(run, selection_path)
    assert selection["data_role"] == "validation"
    assert selection["selected"]["step"] == 200
    assert selection["selected"]["mean_turns_successful"] == pytest.approx(3.5)

    export = export_checkpoint(
        selection_path, tmp_path / "exports" / "seed0", dry_run=True
    )
    assert export["status"] == "planned"
    assert export["checkpoint_metadata"]["has_lora_adapter"] is True
    assert "verl.model_merger" in export["commands"]["fsdp_merge"]
    assert (tmp_path / "exports" / "seed0" / "export_manifest.json").is_file()


@pytest.mark.parametrize("adapter_state", ["missing", "incomplete"])
def test_lora_export_fails_closed_without_complete_adapter(
    tmp_path: Path, adapter_state: str
) -> None:
    run = tmp_path / adapter_state
    _write_json(
        run / "manifest.json",
        {
            "method": "no_concat_episode_grpo",
            "environment": "sokoban",
            "seed": 0,
            "advantage_estimator": "no_concat_episode_grpo",
            "validation_n_envs": 1,
            "lora_rank": 4,
        },
    )
    (run / "validation").mkdir(parents=True)
    (run / "validation" / "100.jsonl").write_text(
        json.dumps({"traj_success": True, "num_turns": 2}) + "\n"
    )
    _actor_checkpoint(run, 100, adapter_state=adapter_state)
    selection_path = run / "selection.json"
    select_validation_checkpoint(run, selection_path)
    with pytest.raises(ValueError, match="LoRA|adapter"):
        plan_checkpoint_export(selection_path, tmp_path / "export")


def test_rank_zero_export_allows_checkpoint_without_adapter(tmp_path: Path) -> None:
    run = tmp_path / "rank0"
    _write_json(
        run / "manifest.json",
        {
            "method": "concat_grpo",
            "environment": "sokoban",
            "seed": 0,
            "advantage_estimator": "grpo",
            "validation_n_envs": 1,
            "lora_rank": 0,
        },
    )
    (run / "validation").mkdir(parents=True)
    (run / "validation" / "100.jsonl").write_text(
        json.dumps({"traj_success": True, "num_turns": 2}) + "\n"
    )
    _actor_checkpoint(run, 100, adapter_state="missing")
    selection_path = run / "selection.json"
    select_validation_checkpoint(run, selection_path)
    plan = plan_checkpoint_export(selection_path, tmp_path / "export")
    assert plan["lora_rank"] == 0
    assert plan["checkpoint_metadata"]["has_lora_adapter"] is False
    assert plan["commands"]["lora_merge"] is None


def _training_run(root: Path, train_seed: int) -> Path:
    run = root / "training" / f"seed{train_seed}"
    _write_json(
        run / "manifest.json",
        {
            "commit": "parent",
            "verl_commit": "submodule",
            "git_dirty": False,
            "method": "no_concat_episode_grpo",
            "environment": "sokoban",
            "seed": train_seed,
            "advantage_estimator": "no_concat_episode_grpo",
            "concat_multi_turn": False,
            "total_steps": 200,
            "validation_n_envs": 2,
        },
    )
    (run / "train_command.sh").write_text("true\n")
    (run / "resolved_config.yaml").write_text("resolved\n")
    validation = run / "validation" / "200.jsonl"
    validation.parent.mkdir(parents=True, exist_ok=True)
    validation.write_text(
        json.dumps({"traj_success": True, "num_turns": 3})
        + "\n"
        + json.dumps({"traj_success": False, "num_turns": 4})
        + "\n"
    )
    _write_json(
        run / "gpu_metrics" / "gpu_summary.json",
        {
            "return_code": 0,
            "sample_count": 2,
            "peak_vram_mib": 1000 + train_seed,
            "gpu_hours": 10.0 + train_seed,
        },
    )
    _write_json(
        run / "parity.json",
        {
            "gate_enabled": True,
            "gate_passed": True,
            "metrics": {
                "ratio_p95": 1.01 + train_seed * 0.01,
                "ratio_p99": 1.02 + train_seed * 0.01,
                "mean_abs_logprob_delta": 0.01 + train_seed * 0.001,
            },
        },
    )
    return run


def _final_run(root: Path, train_seed: int) -> Path:
    run = root / "final" / f"seed{train_seed}"
    model = root / "models" / f"seed{train_seed}"
    selection = root / "selections" / f"seed{train_seed}.json"
    export = root / "exports" / f"seed{train_seed}.json"
    training_run = _training_run(root, train_seed)
    _write_json(selection, {"artifact_type": "validation_checkpoint_selection"})
    _write_json(
        export,
        {
            "artifact_type": "fsdp_lora_checkpoint_export",
            "status": "complete",
            "model_path": str(model),
            "environment": "sokoban",
            "method": "no_concat_episode_grpo",
            "train_seed": train_seed,
            "checkpoint_step": 200,
            "source_run_dir": str(training_run),
        },
    )
    _write_json(
        run / "manifest.json",
        {
            "commit": "parent",
            "verl_commit": "submodule",
            "git_dirty": False,
            "method": "no_concat_episode_grpo",
            "environment": "sokoban",
            "model": str(model),
            "evaluation_role": "final_test",
            "observation_ablation": "none",
            "concat_multi_turn": False,
            "n_envs": 2,
            "seed_start": 10129,
            "source_run_dir": str(training_run),
            "source_selection_manifest": str(selection),
            "source_export_manifest": str(export),
            "source_method": "no_concat_episode_grpo",
            "source_environment": "sokoban",
            "source_train_seed": train_seed,
            "source_checkpoint_step": 200,
        },
    )
    (run / "eval_command.sh").write_text("true\n")
    (run / "resolved_config.txt").write_text("resolved\n")
    _write_json(
        run / "gpu_metrics" / "gpu_summary.json",
        {
            "return_code": 0,
            "sample_count": 2,
            "peak_vram_mib": 100 + train_seed,
            "gpu_hours": 0.5,
        },
    )
    for index, success in enumerate((True, train_seed != 2)):
        episode = run / "episodes" / str(index)
        _write_json(
            episode / "metrics.json",
            {
                "rollout_id": f"{train_seed}-{index}",
                "seed": 10129 + index,
                "success": success,
                "num_turns": 2 + index,
                "cumulative_reward": float(success),
                "finish_reason": "done" if success else "max_turns",
                "env_name": "Sokoban",
                "observation_ablation": "none",
            },
        )
        _write_json(episode / "assistant_texts.json", ["<answer>right</answer>"])
        (episode / "transcript.txt").write_text("synthetic")
    return run


def _base_run(root: Path) -> Path:
    run = root / "base"
    _write_json(
        run / "manifest.json",
        {
            "commit": "parent",
            "verl_commit": "submodule",
            "git_dirty": False,
            "method": "base",
            "environment": "sokoban",
            "model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "evaluation_role": "base_eval",
            "observation_ablation": "none",
            "n_envs": 2,
            "seed_start": 10129,
        },
    )
    (run / "eval_command.sh").write_text("true\n")
    (run / "resolved_config.txt").write_text("resolved\n")
    _write_json(
        run / "gpu_metrics" / "gpu_summary.json",
        {
            "return_code": 0,
            "sample_count": 2,
            "peak_vram_mib": 500,
            "gpu_hours": 0.25,
        },
    )
    for index, success in enumerate((True, False)):
        episode = run / "episodes" / str(index)
        _write_json(
            episode / "metrics.json",
            {
                "rollout_id": f"base-{index}",
                "seed": 10129 + index,
                "success": success,
                "num_turns": 2 + index,
                "cumulative_reward": float(success),
                "finish_reason": "done" if success else "max_turns",
                "env_name": "Sokoban",
                "observation_ablation": "none",
            },
        )
        _write_json(episode / "assistant_texts.json", ["<answer>right</answer>"])
        (episode / "transcript.txt").write_text("synthetic")
    return run


def test_final_results_require_linked_three_seed_final_tests(tmp_path: Path) -> None:
    runs = [_final_run(tmp_path, seed) for seed in range(3)]
    result = aggregate_final_tests(
        runs,
        tmp_path / "results",
        base_run_dirs=[_base_run(tmp_path)],
        expected_methods=("no_concat_episode_grpo",),
        expected_environments=("sokoban",),
    )
    aggregate = result["aggregates"][0]
    assert aggregate["Status"] == "complete"
    assert aggregate["N Seeds"] == 3
    assert aggregate["Visual Success"] == pytest.approx(5 / 6)
    assert aggregate["GPU·h"] == pytest.approx(11.0)
    assert aggregate["GPU·h Total"] == pytest.approx(33.0)
    assert aggregate["Peak VRAM"] == pytest.approx(1002.0)
    assert aggregate["Ratio P95"] == pytest.approx(1.03)
    assert aggregate["Ratio P99"] == pytest.approx(1.04)
    assert aggregate["Mean Abs Logprob Delta"] == pytest.approx(0.012)
    assert aggregate["Parity Gate Passed"] is True
    assert result["per_run"][0]["Evaluation GPU·h"] == pytest.approx(0.5)
    assert result["per_run"][0]["GPU·h"] == pytest.approx(10.0)
    assert (tmp_path / "results" / "main_results.csv").is_file()
    assert result["registry_complete"] is True
    assert len(result["main_results"]) == 2

    published = tmp_path / "published.csv"
    published.write_text("placeholder\n")
    publication = publish_final_results(tmp_path / "results", published)
    assert publication["row_count"] == 2
    assert published.read_text() == (tmp_path / "results" / "main_results.csv").read_text()
    assert (
        tmp_path / "results" / "pre_publish_main_results.csv"
    ).read_text() == "placeholder\n"

    incomplete = aggregate_final_tests(
        runs[:2],
        tmp_path / "incomplete",
        base_run_dirs=[tmp_path / "base"],
        expected_methods=("no_concat_episode_grpo",),
        expected_environments=("sokoban",),
    )
    assert incomplete["aggregates"][0]["Status"] == "incomplete-artifacts"
    assert incomplete["aggregates"][0]["Visual Success"] is None
    assert incomplete["registry_complete"] is False


def test_repository_experiment_contract_is_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    report = validate_experiment_contract(
        root / "experiments" / "matrix.yaml", repo_root=root
    )
    assert report["valid"] is True
    assert len(report["checked_partitions"]) == 6


def test_documented_gpu_budget_includes_each_frozen_checkpoint_final_test() -> None:
    """The planning budget must charge every frozen checkpoint its own final test.

    The contract is over the planning documents that own compute budgets.
    README.md and RESUME_PROJECT_CN.md describe the research and its results,
    not the rental plan, so they are deliberately not required to restate these
    figures.
    """
    root = Path(__file__).resolve().parents[2]
    common = 0.5 + 2.0 + 98.0 * 50 / 401 + 9 * 12.8 * 50 / 401
    winner_low = common + 96.0 + 3 * (0.8 + 1.2) + 3.0
    winner_high = common + 96.0 + 3 * (0.8 + 1.2) + 6.0
    full_low = common + 294.0 + 9 * (0.8 + 1.2) + 3.0
    full_high = common + 294.0 + 9 * (0.8 + 1.2) + 6.0
    assert winner_low == pytest.approx(134.08, abs=0.01)
    assert winner_high == pytest.approx(137.08, abs=0.01)
    assert full_low == pytest.approx(344.08, abs=0.01)
    assert full_high == pytest.approx(347.08, abs=0.01)

    expected_ranges = {
        "PREDICTED_METRICS.md": ("134–137", "344–347"),
        "EXPERIMENTS.md": ("134–137", "344–347"),
        "GPU_EXECUTION_CHECKLIST.md": ("134–137", "344–347"),
        "USER_ACTIONS.md": ("134–137", "344–347"),
        "PROJECT_STATUS.md": ("134–137", "344–347"),
    }
    for relative_path, ranges in expected_ranges.items():
        document = (root / relative_path).read_text()
        for expected_range in ranges:
            assert expected_range in document, relative_path


def test_matrix_final_test_dry_run_writes_linked_manifest(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    selection = tmp_path / "selection.json"
    export = tmp_path / "export.json"
    model = tmp_path / "model"
    training_run = tmp_path / "training_seed0"
    _write_json(selection, {"artifact_type": "validation_checkpoint_selection"})
    _write_json(
        export,
        {
            "artifact_type": "fsdp_lora_checkpoint_export",
            "status": "planned",
            "environment": "sokoban",
            "method": "no_concat_episode_grpo",
            "train_seed": 0,
            "checkpoint_step": 200,
            "model_path": str(model),
            "source_run_dir": str(training_run),
            "selection_manifest": str(selection),
        },
    )
    final_root = tmp_path / "final"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "DRY_RUN": "1",
            "FINAL_TEST_ROOT": str(final_root),
        }
    )
    result = subprocess.run(
        ["bash", "scripts/run_experiment_matrix.sh", "final-test", str(export)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifests = list(final_root.rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["evaluation_role"] == "final_test"
    assert manifest["source_train_seed"] == 0
    assert manifest["source_export_manifest"] == str(export)
