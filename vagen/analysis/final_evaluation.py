"""Leakage-safe checkpoint selection, export, and final-test aggregation.

The commands in this module deliberately separate three data roles:

* validation JSONL is used only to select a checkpoint;
* an exported Hugging Face model is linked to that selection;
* final results are computed only from evaluation runs whose manifest declares
  ``evaluation_role=final_test``.

All planning and validation paths are CPU-only.  The expensive checkpoint merge
is performed only by ``export`` without ``--dry-run``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np

from vagen.analysis.analyze_rollouts import (
    RESULT_COLUMNS,
    analyze_training_rows,
    build_result_row,
    collect_evaluation_episodes,
    load_training_rows,
)


SELECTION_ARTIFACT = "validation_checkpoint_selection"
EXPORT_ARTIFACT = "fsdp_lora_checkpoint_export"
FINAL_TEST_ROLE = "final_test"


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except OSError as error:
        raise ValueError(f"cannot read JSON object: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_object(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_jsonl_files(directory: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in directory.glob("*.jsonl"):
        if path.stem.isdigit() and path.is_file():
            files.append((int(path.stem), path))
    return sorted(files)


def select_validation_checkpoint(
    run_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Select a saved checkpoint using complete validation artifacts only."""
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = _read_object(manifest_path)
    if manifest.get("evaluation_role") is not None:
        raise ValueError("checkpoint selection requires a training run, not an eval run")
    if not manifest.get("advantage_estimator"):
        raise ValueError("training manifest is missing advantage_estimator")

    expected_episodes = manifest.get("validation_n_envs")
    if not isinstance(expected_episodes, int) or expected_episodes <= 0:
        raise ValueError("training manifest has invalid validation_n_envs")

    validation_dir = (run_dir / "validation").resolve()
    if not validation_dir.is_dir():
        raise ValueError(f"validation directory not found: {validation_dir}")

    candidates: list[dict[str, Any]] = []
    for step, validation_path in _numeric_jsonl_files(validation_dir):
        actor_dir = run_dir / "checkpoints" / f"global_step_{step}" / "actor"
        analysis = analyze_training_rows(load_training_rows([validation_path]))
        actual_episodes = int(analysis["trajectories"])
        success_rate = analysis["success_rate"]
        mean_turns = analysis["mean_turns_successful"]
        issues: list[str] = []
        if actual_episodes != expected_episodes:
            issues.append(
                f"expected {expected_episodes} trajectories, found {actual_episodes}"
            )
        if success_rate is None:
            issues.append("success_rate is missing")
        if not actor_dir.is_dir():
            issues.append(f"saved actor checkpoint is missing for step {step}")
        candidates.append(
            {
                "step": step,
                "validation_path": str(validation_path.resolve()),
                "validation_sha256": _sha256(validation_path),
                "checkpoint_actor_dir": str(actor_dir.resolve()),
                "expected_trajectories": expected_episodes,
                "actual_trajectories": actual_episodes,
                "success_rate": success_rate,
                "mean_turns_successful": mean_turns,
                "eligible": not issues,
                "issues": issues,
            }
        )

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise ValueError(
            "no complete validation artifact has a matching saved actor checkpoint"
        )

    def selection_key(candidate: dict[str, Any]) -> tuple[float, float, int]:
        turns = candidate["mean_turns_successful"]
        turn_tiebreaker = -float(turns) if turns is not None else -math.inf
        return float(candidate["success_rate"]), turn_tiebreaker, int(candidate["step"])

    selected = max(eligible, key=selection_key)
    payload = {
        "schema_version": 1,
        "artifact_type": SELECTION_ARTIFACT,
        "data_role": "validation",
        "selection_policy": [
            "max_success_rate",
            "min_mean_turns_successful",
            "max_checkpoint_step",
        ],
        "source_run_dir": str(run_dir),
        "source_run_manifest": str(manifest_path.resolve()),
        "source_run_manifest_sha256": _sha256(manifest_path),
        "environment": manifest.get("environment"),
        "method": manifest.get("method"),
        "train_seed": manifest.get("seed"),
        "expected_validation_trajectories": expected_episodes,
        "candidates": candidates,
        "selected": selected,
    }
    _write_object(output_path, payload)
    return payload


def _validate_fsdp_actor(actor_dir: Path) -> dict[str, Any]:
    config_path = actor_dir / "fsdp_config.json"
    hf_config_path = actor_dir / "huggingface" / "config.json"
    if not config_path.is_file():
        raise ValueError(f"missing FSDP metadata: {config_path}")
    if not hf_config_path.is_file():
        raise ValueError(f"missing Hugging Face config: {hf_config_path}")
    config = _read_object(config_path)
    world_size = config.get("world_size")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"invalid world_size in {config_path}")
    shards = [
        actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
    ]
    missing = [path for path in shards if not path.is_file()]
    if missing:
        raise ValueError(
            "missing FSDP model shard(s): " + ", ".join(str(path) for path in missing)
        )
    adapter_dir = actor_dir / "lora_adapter"
    adapter_files = (
        adapter_dir / "adapter_config.json",
        adapter_dir / "adapter_model.safetensors",
    )
    if adapter_dir.exists() and not all(path.is_file() for path in adapter_files):
        raise ValueError(f"incomplete LoRA adapter checkpoint: {adapter_dir}")
    return {
        "world_size": world_size,
        "model_shards": [str(path.resolve()) for path in shards],
        "has_lora_adapter": all(path.is_file() for path in adapter_files),
        "lora_adapter_dir": str(adapter_dir.resolve()),
    }


def _shell_command(arguments: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(argument)) for argument in arguments)


def plan_checkpoint_export(
    selection_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate an FSDP/LoRA checkpoint and return an executable export plan."""
    selection_path = selection_path.resolve()
    selection = _read_object(selection_path)
    if selection.get("artifact_type") != SELECTION_ARTIFACT:
        raise ValueError(f"not a checkpoint-selection artifact: {selection_path}")
    if selection.get("data_role") != "validation":
        raise ValueError("checkpoint selection was not derived exclusively from validation")
    selected = selection.get("selected")
    if not isinstance(selected, dict) or not selected.get("eligible"):
        raise ValueError("selection artifact has no eligible selected checkpoint")

    source_manifest_value = selection.get("source_run_manifest")
    if not source_manifest_value:
        raise ValueError("selection artifact is missing source_run_manifest")
    source_manifest_path = Path(str(source_manifest_value)).resolve()
    if not source_manifest_path.is_file():
        raise ValueError(f"source training manifest is missing: {source_manifest_path}")
    expected_source_sha = selection.get("source_run_manifest_sha256")
    if not isinstance(expected_source_sha, str) or _sha256(
        source_manifest_path
    ) != expected_source_sha:
        raise ValueError("source training manifest changed after checkpoint selection")
    source_manifest = _read_object(source_manifest_path)
    lora_rank = source_manifest.get("lora_rank")
    if isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or lora_rank < 0:
        raise ValueError("source training manifest has invalid lora_rank")

    actor_dir = Path(str(selected["checkpoint_actor_dir"])).resolve()
    actor_metadata = _validate_fsdp_actor(actor_dir)
    if lora_rank > 0 and not actor_metadata["has_lora_adapter"]:
        raise ValueError(
            f"LoRA rank {lora_rank} training checkpoint has no complete adapter: "
            f"{actor_dir / 'lora_adapter'}"
        )
    output_dir = output_dir.resolve()
    fsdp_hf_dir = output_dir / "fsdp_hf"
    model_dir = output_dir / "model"
    merge_command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(fsdp_hf_dir),
    ]
    lora_command = [
        sys.executable,
        "-m",
        "vagen.analysis.final_evaluation",
        "merge-lora",
        "--base-model",
        str(fsdp_hf_dir),
        "--adapter",
        actor_metadata["lora_adapter_dir"],
        "--output-dir",
        str(model_dir),
    ]
    return {
        "schema_version": 1,
        "artifact_type": EXPORT_ARTIFACT,
        "status": "planned",
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "source_run_dir": selection.get("source_run_dir"),
        "source_run_manifest": selection.get("source_run_manifest"),
        "environment": selection.get("environment"),
        "method": selection.get("method"),
        "train_seed": selection.get("train_seed"),
        "checkpoint_step": selected.get("step"),
        "checkpoint_actor_dir": str(actor_dir),
        "lora_rank": lora_rank,
        "checkpoint_metadata": actor_metadata,
        "fsdp_hf_dir": str(fsdp_hf_dir),
        "model_path": str(model_dir),
        "commands": {
            "fsdp_merge": merge_command,
            "fsdp_merge_shell": _shell_command(merge_command),
            "lora_merge": lora_command if actor_metadata["has_lora_adapter"] else None,
            "lora_merge_shell": (
                _shell_command(lora_command)
                if actor_metadata["has_lora_adapter"]
                else None
            ),
        },
    }


def _verify_exported_model(model_dir: Path) -> None:
    if not (model_dir / "config.json").is_file():
        raise ValueError(f"exported model is missing config.json: {model_dir}")
    weight_files = list(model_dir.glob("*.safetensors")) + list(
        model_dir.glob("pytorch_model*.bin")
    )
    if not weight_files:
        raise ValueError(f"exported model has no Hugging Face weight files: {model_dir}")


def merge_lora_model(base_model: Path, adapter: Path, output_dir: Path) -> None:
    """Merge a PEFT adapter into a vision-language Hugging Face model on CPU."""
    try:
        from peft import PeftModel
        try:
            from transformers import AutoModelForImageTextToText as AutoVisionModel
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVisionModel
    except ImportError as error:
        raise RuntimeError("transformers and peft are required to merge LoRA") from error

    if output_dir.exists():
        raise ValueError(f"refusing to overwrite export directory: {output_dir}")
    model = AutoVisionModel.from_pretrained(
        base_model,
        device_map="cpu",
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
    for path in base_model.iterdir():
        if path.name.startswith("model") or path.name.startswith("pytorch_model"):
            continue
        if path.name == "lora_adapter":
            continue
        target = output_dir / path.name
        if target.exists():
            continue
        if path.is_dir():
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)


def export_checkpoint(
    selection_path: Path,
    output_dir: Path,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Plan or execute FSDP conversion and optional LoRA merge."""
    plan = plan_checkpoint_export(selection_path, output_dir)
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "export_manifest.json"
    if dry_run:
        plan["dry_run"] = True
        _write_object(manifest_path, plan)
        return plan

    fsdp_hf_dir = Path(plan["fsdp_hf_dir"])
    model_dir = Path(plan["model_path"])
    if fsdp_hf_dir.exists() or model_dir.exists():
        raise ValueError(
            "refusing to overwrite an existing export; choose a new output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    pythonpath = [str(repo_root), str(repo_root / "verl")]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    subprocess.run(
        plan["commands"]["fsdp_merge"],
        cwd=repo_root,
        env=environment,
        check=True,
    )

    metadata = plan["checkpoint_metadata"]
    if metadata["has_lora_adapter"]:
        generated_adapter = fsdp_hf_dir / "lora_adapter"
        source_adapter = Path(metadata["lora_adapter_dir"])
        generated_adapter.mkdir(parents=True, exist_ok=True)
        for adapter_file in (
            "adapter_config.json",
            "adapter_model.safetensors",
        ):
            shutil.copy2(source_adapter / adapter_file, generated_adapter / adapter_file)
        merge_lora_model(fsdp_hf_dir, generated_adapter, model_dir)
    else:
        shutil.copytree(fsdp_hf_dir, model_dir)
    _verify_exported_model(model_dir)
    plan["status"] = "complete"
    plan["dry_run"] = False
    _write_object(manifest_path, plan)
    return plan


def _final_test_record(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = _read_object(manifest_path)
    if manifest.get("evaluation_role") != FINAL_TEST_ROLE:
        raise ValueError(
            f"{run_dir} is not a final-test run (evaluation_role={manifest.get('evaluation_role')!r})"
        )
    if manifest.get("observation_ablation") != "none":
        raise ValueError(f"final result cannot use an observation ablation: {run_dir}")

    linkage_issues: list[str] = []
    required = (
        "source_run_dir",
        "source_selection_manifest",
        "source_export_manifest",
        "source_method",
        "source_train_seed",
        "source_checkpoint_step",
    )
    for key in required:
        if manifest.get(key) in {None, ""}:
            linkage_issues.append(f"manifest is missing {key}")
    if manifest.get("source_environment", manifest.get("environment")) != manifest.get(
        "environment"
    ):
        linkage_issues.append("source and evaluation environments differ")

    export_path_value = manifest.get("source_export_manifest")
    if export_path_value:
        export_path = Path(str(export_path_value))
        try:
            export_manifest = _read_object(export_path)
            if export_manifest.get("artifact_type") != EXPORT_ARTIFACT:
                linkage_issues.append("source_export_manifest has the wrong artifact type")
            if export_manifest.get("status") != "complete":
                linkage_issues.append("source export is not complete")
            if Path(str(export_manifest.get("model_path"))).resolve() != Path(
                str(manifest.get("model"))
            ).resolve():
                linkage_issues.append("evaluated model does not match exported model")
            for field, source_field in (
                ("environment", "environment"),
                ("source_method", "method"),
                ("source_train_seed", "train_seed"),
                ("source_checkpoint_step", "checkpoint_step"),
                ("source_run_dir", "source_run_dir"),
            ):
                if manifest.get(field) != export_manifest.get(source_field):
                    linkage_issues.append(
                        f"evaluation/export linkage mismatch for {field}"
                    )
        except ValueError as error:
            linkage_issues.append(str(error))

    source_row: dict[str, Any] = {}
    source_parity_metrics: dict[str, Any] = {}
    source_parity_gate: bool | None = None
    source_run_value = manifest.get("source_run_dir")
    if source_run_value:
        source_run = Path(str(source_run_value)).resolve()
        try:
            source_manifest = _read_object(source_run / "manifest.json")
            for evaluation_field, training_field in (
                ("source_method", "method"),
                ("source_train_seed", "seed"),
                ("source_environment", "environment"),
            ):
                if manifest.get(evaluation_field) != source_manifest.get(training_field):
                    linkage_issues.append(
                        f"evaluation/training linkage mismatch for {evaluation_field}"
                    )
            source_row = build_result_row(source_run)
            if source_row["Status"] != "complete":
                linkage_issues.append(
                    f"source training run is {source_row['Status']}, not complete"
                )
            source_parity = _read_object(source_run / "parity.json")
            metrics_value = source_parity.get("metrics")
            if isinstance(metrics_value, dict):
                source_parity_metrics = metrics_value
            source_parity_gate = source_parity.get("gate_passed")
            if source_parity_gate is not True:
                linkage_issues.append("source parity gate did not pass")
            for metric in (
                "ratio_p95",
                "ratio_p99",
                "mean_abs_logprob_delta",
            ):
                value = source_parity_metrics.get(metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    linkage_issues.append(f"source parity metric {metric} is missing or invalid")
        except ValueError as error:
            linkage_issues.append(str(error))

    episodes = collect_evaluation_episodes(run_dir)
    expected_episodes = manifest.get("n_envs")
    if not isinstance(expected_episodes, int) or expected_episodes <= 0:
        linkage_issues.append("manifest has invalid n_envs")
    elif len(episodes) != expected_episodes:
        linkage_issues.append(
            f"expected {expected_episodes} final-test episodes, found {len(episodes)}"
        )
    if len({episode.get("seed") for episode in episodes}) != len(episodes):
        linkage_issues.append("final-test episode seeds are not unique")

    base_row = build_result_row(run_dir)
    status = base_row["Status"]
    if linkage_issues and status == "complete":
        status = "incomplete-artifacts"
    successes = sum(bool(episode["success"]) for episode in episodes)
    record = {
        **base_row,
        "Status": status,
        "Method": manifest.get("source_method"),
        "Environment": manifest.get("environment"),
        "Seed": manifest.get("source_train_seed"),
        "Peak VRAM": source_row.get("Peak VRAM"),
        "GPU·h": source_row.get("GPU·h"),
        "Ratio P95": source_row.get("Ratio P95"),
        "Ratio P99": source_parity_metrics.get("ratio_p99"),
        "Mean Abs Logprob Delta": source_parity_metrics.get(
            "mean_abs_logprob_delta"
        ),
        "Parity Gate Passed": source_parity_gate,
        "Evaluation Peak VRAM": base_row.get("Peak VRAM"),
        "Evaluation GPU·h": base_row.get("GPU·h"),
        "Evaluation Role": FINAL_TEST_ROLE,
        "Training Run": manifest.get("source_run_dir"),
        "Checkpoint Step": manifest.get("source_checkpoint_step"),
        "Selection Manifest": manifest.get("source_selection_manifest"),
        "Export Manifest": manifest.get("source_export_manifest"),
        "Evaluation Run": str(run_dir),
        "Episode Count": len(episodes),
        "Successful Episodes": successes,
        "Integrity Issues": "; ".join(linkage_issues),
    }
    return record, episodes


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    z = NormalDist().inv_cdf(0.975)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(
        (proportion * (1 - proportion) + z * z / (4 * trials)) / trials
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _base_eval_aggregate(run_dir: Path) -> dict[str, Any]:
    manifest = _read_object(run_dir / "manifest.json")
    if manifest.get("evaluation_role") != "base_eval":
        raise ValueError(f"{run_dir} is not a base_eval run")
    if manifest.get("observation_ablation") != "none":
        raise ValueError(f"base result cannot use an observation ablation: {run_dir}")
    row = build_result_row(run_dir)
    episodes = collect_evaluation_episodes(run_dir)
    successes = sum(bool(episode["success"]) for episode in episodes)
    ci_low, ci_high = _wilson_interval(successes, len(episodes))
    return {
        "Method": "base",
        "Environment": manifest.get("environment"),
        "Train Seeds": "",
        "N Seeds": 0,
        "Visual Success": row["Visual Success"],
        "Success Macro Mean": row["Visual Success"],
        "Success Seed Std": None,
        "Success Wilson 95% Low": ci_low if episodes else None,
        "Success Wilson 95% High": ci_high if episodes else None,
        "Successful Episodes": successes,
        "Episode Count": len(episodes),
        "Mean Turns": row["Mean Turns"],
        "Peak VRAM": row["Peak VRAM"],
        "GPU·h": row["GPU·h"],
        "GPU·h Total": row["GPU·h"],
        "Ratio P95": None,
        "Max Ratio P95 Deviation": None,
        "Ratio P99": None,
        "Max Ratio P99 Deviation": None,
        "Mean Abs Logprob Delta": None,
        "Parity Gate Passed": None,
        "Status": row["Status"],
        "Commit": row["Commit"],
        "Evidence": str(run_dir.resolve()),
        "Integrity Issues": "",
    }


def _missing_result(environment: str, method: str, issue: str) -> dict[str, Any]:
    return {
        "Method": method,
        "Environment": environment,
        "Train Seeds": "",
        "N Seeds": 0,
        "Visual Success": None,
        "Success Macro Mean": None,
        "Success Seed Std": None,
        "Success Wilson 95% Low": None,
        "Success Wilson 95% High": None,
        "Successful Episodes": None,
        "Episode Count": None,
        "Mean Turns": None,
        "Peak VRAM": None,
        "GPU·h": None,
        "GPU·h Total": None,
        "Ratio P95": None,
        "Max Ratio P95 Deviation": None,
        "Ratio P99": None,
        "Max Ratio P99 Deviation": None,
        "Mean Abs Logprob Delta": None,
        "Parity Gate Passed": None,
        "Status": "incomplete-artifacts",
        "Commit": "",
        "Evidence": "",
        "Integrity Issues": issue,
    }


def aggregate_final_tests(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    expected_train_seeds: tuple[int, ...] = (0, 1, 2),
    base_run_dirs: list[Path] | None = None,
    expected_methods: tuple[str, ...] | None = None,
    expected_environments: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Aggregate complete final tests by environment and method.

    Validation metrics never contribute to final-test success or turn counts.
    Source-training artifacts are checked only for completion, resource use, and
    parity. Incomplete groups are reported with null aggregate fields.
    """
    if not run_dirs:
        raise ValueError("at least one final-test run is required")
    if len(set(expected_train_seeds)) != len(expected_train_seeds):
        raise ValueError("expected_train_seeds contains duplicates")

    records: list[dict[str, Any]] = []
    episodes_by_run: dict[str, list[dict[str, Any]]] = {}
    for run_dir in run_dirs:
        record, episodes = _final_test_record(run_dir)
        records.append(record)
        episodes_by_run[record["Evaluation Run"]] = episodes

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["Environment"]), str(record["Method"]))].append(record)

    aggregates: list[dict[str, Any]] = []
    expected_set = set(expected_train_seeds)
    for (environment, method), group in sorted(grouped.items()):
        seeds = [record["Seed"] for record in group]
        seed_set = {seed for seed in seeds if isinstance(seed, int)}
        issues: list[str] = []
        if len(seeds) != len(seed_set):
            issues.append("duplicate or non-integer training seed")
        if seed_set != expected_set:
            issues.append(
                f"expected training seeds {sorted(expected_set)}, found {sorted(seed_set)}"
            )
        non_complete = [
            record["Evaluation Run"]
            for record in group
            if record["Status"] != "complete"
        ]
        if non_complete:
            issues.append("incomplete final-test run(s): " + ", ".join(non_complete))

        complete = not issues
        all_episodes = [
            episode
            for record in group
            for episode in episodes_by_run[record["Evaluation Run"]]
        ]
        successful = [episode for episode in all_episodes if episode["success"]]
        per_seed_rates = [
            record["Successful Episodes"] / record["Episode Count"]
            for record in group
            if record["Episode Count"]
        ]
        successes = len(successful)
        trials = len(all_episodes)
        ci_low, ci_high = _wilson_interval(successes, trials)
        aggregates.append(
            {
                "Method": method,
                "Environment": environment,
                "Train Seeds": ",".join(str(seed) for seed in sorted(seed_set)),
                "N Seeds": len(seed_set),
                "Visual Success": successes / trials if complete and trials else None,
                "Success Macro Mean": (
                    float(np.mean(per_seed_rates))
                    if complete and per_seed_rates
                    else None
                ),
                "Success Seed Std": (
                    float(np.std(per_seed_rates, ddof=1))
                    if complete and len(per_seed_rates) > 1
                    else (0.0 if complete and per_seed_rates else None)
                ),
                "Success Wilson 95% Low": ci_low if complete else None,
                "Success Wilson 95% High": ci_high if complete else None,
                "Successful Episodes": successes if complete else None,
                "Episode Count": trials if complete else None,
                "Mean Turns": (
                    float(np.mean([episode["num_turns"] for episode in successful]))
                    if complete and successful
                    else None
                ),
                "Peak VRAM": (
                    max(float(record["Peak VRAM"]) for record in group)
                    if complete and all(record["Peak VRAM"] is not None for record in group)
                    else None
                ),
                "GPU·h": (
                    float(np.mean([float(record["GPU·h"]) for record in group]))
                    if complete
                    and all(record["GPU·h"] is not None for record in group)
                    else None
                ),
                "GPU·h Total": (
                    sum(float(record["GPU·h"]) for record in group)
                    if complete and all(record["GPU·h"] is not None for record in group)
                    else None
                ),
                "Ratio P95": (
                    max(
                        (float(record["Ratio P95"]) for record in group),
                        key=lambda ratio: abs(ratio - 1.0),
                    )
                    if complete
                    and all(record["Ratio P95"] is not None for record in group)
                    else None
                ),
                "Max Ratio P95 Deviation": (
                    max(abs(float(record["Ratio P95"]) - 1.0) for record in group)
                    if complete
                    and all(record["Ratio P95"] is not None for record in group)
                    else None
                ),
                "Ratio P99": (
                    max(
                        (float(record["Ratio P99"]) for record in group),
                        key=lambda ratio: abs(ratio - 1.0),
                    )
                    if complete
                    and all(record["Ratio P99"] is not None for record in group)
                    else None
                ),
                "Max Ratio P99 Deviation": (
                    max(abs(float(record["Ratio P99"]) - 1.0) for record in group)
                    if complete
                    and all(record["Ratio P99"] is not None for record in group)
                    else None
                ),
                "Mean Abs Logprob Delta": (
                    max(float(record["Mean Abs Logprob Delta"]) for record in group)
                    if complete
                    and all(
                        record["Mean Abs Logprob Delta"] is not None
                        for record in group
                    )
                    else None
                ),
                "Parity Gate Passed": (
                    all(record["Parity Gate Passed"] is True for record in group)
                    if complete
                    else None
                ),
                "Status": "complete" if complete else "incomplete-artifacts",
                "Commit": ",".join(
                    sorted({str(record["Commit"]) for record in group if record["Commit"]})
                ),
                "Evidence": ";".join(record["Evaluation Run"] for record in group),
                "Integrity Issues": "; ".join(issues),
            }
        )

    actual_methods = {method for _, method in grouped}
    actual_environments = {environment for environment, _ in grouped}
    expected_method_set = set(expected_methods or tuple(sorted(actual_methods)))
    expected_environment_set = set(
        expected_environments or tuple(sorted(actual_environments))
    )
    for environment in sorted(expected_environment_set):
        for method in sorted(expected_method_set):
            if (environment, method) not in grouped:
                aggregates.append(
                    _missing_result(
                        environment,
                        method,
                        "missing required environment/method final-test group",
                    )
                )
    for environment, method in sorted(grouped):
        if environment not in expected_environment_set or method not in expected_method_set:
            for aggregate in aggregates:
                if (
                    aggregate["Environment"] == environment
                    and aggregate["Method"] == method
                ):
                    for field in (
                        "Visual Success",
                        "Success Macro Mean",
                        "Success Seed Std",
                        "Success Wilson 95% Low",
                        "Success Wilson 95% High",
                        "Successful Episodes",
                        "Episode Count",
                        "Mean Turns",
                        "Peak VRAM",
                        "GPU·h",
                        "GPU·h Total",
                        "Ratio P95",
                        "Max Ratio P95 Deviation",
                        "Ratio P99",
                        "Max Ratio P99 Deviation",
                        "Mean Abs Logprob Delta",
                        "Parity Gate Passed",
                    ):
                        aggregate[field] = None
                    aggregate["Status"] = "incomplete-artifacts"
                    aggregate["Integrity Issues"] = (
                        str(aggregate["Integrity Issues"])
                        + "; unexpected environment/method group"
                    ).strip("; ")

    base_by_environment: dict[str, dict[str, Any]] = {}
    for base_run in base_run_dirs or []:
        base_result = _base_eval_aggregate(base_run.resolve())
        environment = str(base_result["Environment"])
        if environment in base_by_environment:
            raise ValueError(f"duplicate base_eval run for environment {environment}")
        base_by_environment[environment] = base_result
    base_results = []
    for environment in sorted(expected_environment_set):
        base_results.append(
            base_by_environment.get(
                environment,
                _missing_result(environment, "base", "missing required base_eval run"),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_fields = list(RESULT_COLUMNS) + [
        "Evaluation Role",
        "Training Run",
        "Checkpoint Step",
        "Selection Manifest",
        "Export Manifest",
        "Evaluation Run",
        "Episode Count",
        "Successful Episodes",
        "Evaluation Peak VRAM",
        "Evaluation GPU·h",
        "Ratio P99",
        "Mean Abs Logprob Delta",
        "Parity Gate Passed",
        "Integrity Issues",
    ]
    aggregate_fields = [
        "Method",
        "Environment",
        "Train Seeds",
        "N Seeds",
        "Visual Success",
        "Success Macro Mean",
        "Success Seed Std",
        "Success Wilson 95% Low",
        "Success Wilson 95% High",
        "Successful Episodes",
        "Episode Count",
        "Mean Turns",
        "Peak VRAM",
        "GPU·h",
        "GPU·h Total",
        "Ratio P95",
        "Max Ratio P95 Deviation",
        "Ratio P99",
        "Max Ratio P99 Deviation",
        "Mean Abs Logprob Delta",
        "Parity Gate Passed",
        "Status",
        "Commit",
        "Evidence",
        "Integrity Issues",
    ]
    for name, rows, fields in (
        ("final_test_runs.csv", records, run_fields),
        ("final_test_aggregates.csv", aggregates, aggregate_fields),
        ("base_eval_results.csv", base_results, aggregate_fields),
    ):
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    main_results = []
    for row in base_results + sorted(
        aggregates, key=lambda value: (value["Environment"], value["Method"])
    ):
        main_results.append(
            {
                "Method": row["Method"],
                "Visual Success": row["Visual Success"],
                "Peak VRAM": row["Peak VRAM"],
                "GPU·h": row["GPU·h"],
                "Mean Turns": row["Mean Turns"],
                "Ratio P95": row["Ratio P95"],
                "Status": row["Status"],
                "Environment": row["Environment"],
                "Seed": row["Train Seeds"],
                "Commit": row["Commit"],
                "Evidence": row["Evidence"],
            }
        )
    with (output_dir / "main_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(main_results)
    expected_keys = {
        (environment, method)
        for environment in expected_environment_set
        for method in ({"base"} | expected_method_set)
    }
    actual_keys = {
        (str(row["Environment"]), str(row["Method"])) for row in main_results
    }
    registry_complete = (
        actual_keys == expected_keys
        and all(row["Status"] == "complete" for row in main_results)
    )
    payload = {
        "schema_version": 1,
        "data_role": FINAL_TEST_ROLE,
        "expected_train_seeds": list(expected_train_seeds),
        "expected_methods": sorted(expected_method_set),
        "expected_environments": sorted(expected_environment_set),
        "registry_complete": registry_complete,
        "per_run": records,
        "base_results": base_results,
        "aggregates": aggregates,
        "main_results": main_results,
    }
    _write_object(output_dir / "final_test_results.json", payload)
    return payload


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _parse_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("names must be a unique comma-separated list")
    return names


def publish_final_results(results_dir: Path, output_path: Path) -> dict[str, Any]:
    """Publish a complete result registry while preserving the previous file."""
    results_dir = results_dir.resolve()
    payload = _read_object(results_dir / "final_test_results.json")
    if payload.get("registry_complete") is not True:
        raise ValueError("result registry is incomplete; refusing to publish")
    source = results_dir / "main_results.csv"
    if not source.is_file():
        raise ValueError(f"main result table is missing: {source}")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = results_dir / "pre_publish_main_results.csv"
    if output_path.exists() and not backup_path.exists():
        shutil.copy2(output_path, backup_path)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(output_path)
    publication = {
        "schema_version": 1,
        "source_results": str(source),
        "published_results": str(output_path),
        "backup": str(backup_path) if backup_path.exists() else None,
        "row_count": len(payload.get("main_results", [])),
        "expected_methods": payload.get("expected_methods"),
        "expected_environments": payload.get("expected_environments"),
    }
    _write_object(results_dir / "publication_manifest.json", publication)
    return publication


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select", help="select by validation only")
    select_parser.add_argument("--run", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)

    export_parser = subparsers.add_parser("export", help="export FSDP/LoRA checkpoint")
    export_parser.add_argument("--selection", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--dry-run", action="store_true")

    merge_parser = subparsers.add_parser("merge-lora", help=argparse.SUPPRESS)
    merge_parser.add_argument("--base-model", type=Path, required=True)
    merge_parser.add_argument("--adapter", type=Path, required=True)
    merge_parser.add_argument("--output-dir", type=Path, required=True)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="aggregate independent final tests"
    )
    aggregate_parser.add_argument("--run", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--base-run", type=Path, action="append", default=[])
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    aggregate_parser.add_argument(
        "--expected-train-seeds", type=_parse_seeds, default=(0, 1, 2)
    )
    aggregate_parser.add_argument("--expected-methods", type=_parse_names)
    aggregate_parser.add_argument("--expected-environments", type=_parse_names)

    publish_parser = subparsers.add_parser(
        "publish", help="publish a complete main result registry"
    )
    publish_parser.add_argument("--results-dir", type=Path, required=True)
    publish_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "select":
        result = select_validation_checkpoint(args.run, args.output)
    elif args.command == "export":
        result = export_checkpoint(
            args.selection, args.output_dir, dry_run=args.dry_run
        )
    elif args.command == "merge-lora":
        merge_lora_model(args.base_model, args.adapter, args.output_dir)
        result = {"status": "complete", "model_path": str(args.output_dir.resolve())}
    elif args.command == "aggregate":
        result = aggregate_final_tests(
            args.run,
            args.output_dir,
            expected_train_seeds=args.expected_train_seeds,
            base_run_dirs=args.base_run,
            expected_methods=args.expected_methods,
            expected_environments=args.expected_environments,
        )
    elif args.command == "publish":
        result = publish_final_results(args.results_dir, args.output)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
