"""Summarize evaluation dumps and training rollouts without inventing missing metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ANSWER_PATTERN = re.compile(
    r"<(?P<tag>answer|action)\b[^>]*>(?P<body>.*?)</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
RESULT_COLUMNS = [
    "Method",
    "Visual Success",
    "Peak VRAM",
    "GPU·h",
    "Mean Turns",
    "Ratio P95",
    "Status",
    "Environment",
    "Seed",
    "Commit",
    "Evidence",
]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_template(text: Any) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", normalized)
    return normalized


def _answer_template(text: Any) -> str:
    match = ANSWER_PATTERN.search(str(text or ""))
    return _normalize_template(match.group("body") if match else text)


def _concentration(values: Iterable[str]) -> tuple[float | None, float | None]:
    values = [value for value in values if value]
    if not values:
        return None, None
    counts = Counter(values)
    return max(counts.values()) / len(values), len(counts) / len(values)


def _nested_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                found.append(child)
            found.extend(_nested_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_nested_values(child, key))
    return found


def _correlation_and_slope(
    x_values: list[float], y_values: list[float]
) -> tuple[float | None, float | None]:
    if len(x_values) < 2:
        return None, None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None, None
    correlation = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    return correlation, slope


def collect_evaluation_episodes(root: Path) -> list[dict[str, Any]]:
    """Collect episode metrics and assistant texts from evaluation artifacts.

    Recursively scans for metrics.json files and extracts episode-level statistics,
    combining metrics with assistant response texts for downstream analysis.

    Args:
        root: Directory containing evaluation outputs with metrics.json files.

    Returns:
        List of episode dictionaries with success, turns, rewards, and transcripts.
            Each dictionary contains:
            - rollout_id: Episode identifier
            - seed: Random seed for reproducibility
            - success: Boolean task completion flag
            - num_turns: Number of environment interactions
            - cumulative_reward: Total reward across all turns
            - finish_reason: Termination cause (done, max_turns, error, etc.)
            - environment: Environment name
            - tag_id: Optional experiment tag
            - observation_ablation: Visual ablation mode (none, remove_images, etc.)
            - assistant_texts: List of model responses
            - valid_action_count: Number of valid actions
            - action_check_count: Total action validation attempts
            - metrics_path: Path to source metrics.json
            - transcript_path: Path to episode transcript

    Example:
        >>> episodes = collect_evaluation_episodes(Path("outputs/eval_run_001"))
        >>> len(episodes)
        100
        >>> episodes[0]["success"]
        True
        >>> episodes[0]["num_turns"]
        5

    Note:
        Malformed or unreadable metrics.json files are silently skipped to enable
        partial analysis of incomplete evaluation runs.
    """
    episodes: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = _load_json(metrics_path)
        if metrics is None:
            continue
        assistant_path = metrics_path.with_name("assistant_texts.json")
        try:
            assistant_texts = json.loads(assistant_path.read_text())
        except (OSError, json.JSONDecodeError):
            assistant_texts = []
        if not isinstance(assistant_texts, list):
            assistant_texts = []
        valid_flags = [
            bool(value)
            for value in _nested_values(metrics.get("infos", []), "action_is_valid")
        ]
        episodes.append(
            {
                "rollout_id": metrics.get("rollout_id", metrics_path.parent.name),
                "seed": metrics.get("seed"),
                "success": bool(metrics.get("success", False)),
                "num_turns": int(metrics.get("num_turns", 0) or 0),
                "cumulative_reward": float(metrics.get("cumulative_reward", 0.0) or 0.0),
                "finish_reason": metrics.get("finish_reason"),
                "environment": metrics.get("env_name"),
                "tag_id": metrics.get("tag_id"),
                "observation_ablation": metrics.get("observation_ablation", "none"),
                "assistant_texts": [str(text) for text in assistant_texts],
                "valid_action_count": sum(valid_flags),
                "action_check_count": len(valid_flags),
                "metrics_path": str(metrics_path),
                "transcript_path": str(metrics_path.with_name("transcript.txt")),
            }
        )
    return episodes


def analyze_evaluation_episodes(
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute success rate, template concentration, and visual ablation deltas.

    Analyzes evaluation episodes to detect policy quality, response diversity,
    and the impact of visual observations on task performance.

    Args:
        episodes: Episode dictionaries from collect_evaluation_episodes.

    Returns:
        Summary dictionary with success metrics and quality diagnostics:
            - episodes: Total episode count
            - success_rate: Fraction of episodes completing the task
            - mean_turns: Average turns across all episodes
            - mean_turns_successful: Average turns for successful episodes only
            - mean_cumulative_reward: Average total reward
            - invalid_action_fraction: Fraction of actions failing validation
            - action_checks: Total action validation attempts
            - answer_template_concentration: Max frequency of most common answer template
            - unique_answer_fraction: Ratio of unique answer templates to total responses
            - response_template_concentration: Max frequency of most common response template
            - unique_response_fraction: Ratio of unique response templates to total responses
            - reward_turn_correlation: Correlation between turns and cumulative reward
            - reward_per_extra_turn_slope: Linear slope of reward vs turns
            - by_observation_ablation: Per-ablation success rates and turn statistics

    Example:
        >>> episodes = collect_evaluation_episodes(Path("outputs/eval"))
        >>> analysis = analyze_evaluation_episodes(episodes)
        >>> analysis["success_rate"]
        0.73
        >>> analysis["answer_template_concentration"]
        0.15  # Low concentration indicates diverse responses
        >>> analysis["by_observation_ablation"]["none"]["success_rate"]
        0.73
        >>> analysis["by_observation_ablation"]["remove_images"]["success_rate"]
        0.12  # Large drop indicates vision is critical

    Note:
        Template concentration above 0.5 suggests mode collapse (model repeating
        similar responses). Unique fraction below 0.3 indicates limited diversity.
        Negative reward_per_extra_turn_slope suggests efficiency issues.
    """
    count = len(episodes)
    successes = [episode for episode in episodes if episode["success"]]
    responses = [
        text
        for episode in episodes
        for text in episode["assistant_texts"]
    ]
    answer_concentration, unique_answer_fraction = _concentration(
        _answer_template(response) for response in responses
    )
    response_concentration, unique_response_fraction = _concentration(
        _normalize_template(response) for response in responses
    )
    action_checks = sum(episode["action_check_count"] for episode in episodes)
    valid_actions = sum(episode["valid_action_count"] for episode in episodes)
    reward_turn_correlation, reward_per_turn_slope = _correlation_and_slope(
        [float(episode["num_turns"]) for episode in episodes],
        [float(episode["cumulative_reward"]) for episode in episodes],
    )

    by_ablation: dict[str, dict[str, Any]] = {}
    ablations = sorted(
        {str(episode["observation_ablation"]) for episode in episodes}
    )
    for ablation in ablations:
        subset = [
            episode
            for episode in episodes
            if str(episode["observation_ablation"]) == ablation
        ]
        subset_successes = [episode for episode in subset if episode["success"]]
        by_ablation[ablation] = {
            "episodes": len(subset),
            "success_rate": (
                len(subset_successes) / len(subset) if subset else None
            ),
            "mean_turns_successful": (
                float(np.mean([episode["num_turns"] for episode in subset_successes]))
                if subset_successes
                else None
            ),
        }

    return {
        "episodes": count,
        "success_rate": len(successes) / count if count else None,
        "mean_turns": (
            float(np.mean([episode["num_turns"] for episode in episodes]))
            if episodes
            else None
        ),
        "mean_turns_successful": (
            float(np.mean([episode["num_turns"] for episode in successes]))
            if successes
            else None
        ),
        "mean_cumulative_reward": (
            float(np.mean([episode["cumulative_reward"] for episode in episodes]))
            if episodes
            else None
        ),
        "invalid_action_fraction": (
            1.0 - valid_actions / action_checks if action_checks else None
        ),
        "action_checks": action_checks,
        "answer_template_concentration": answer_concentration,
        "unique_answer_fraction": unique_answer_fraction,
        "response_template_concentration": response_concentration,
        "unique_response_fraction": unique_response_fraction,
        "reward_turn_correlation": reward_turn_correlation,
        "reward_per_extra_turn_slope": reward_per_turn_slope,
        "by_observation_ablation": by_ablation,
    }


def representative_failures(
    episodes: list[dict[str, Any]], limit: int = 10
) -> list[dict[str, Any]]:
    """Extract representative failed episodes for manual review.

    Prioritizes non-error failures (clean task failures rather than crashes) and
    selects diverse failure modes by sorting on validity, turns, and reward.

    Args:
        episodes: Episode dictionaries from collect_evaluation_episodes.
        limit: Maximum number of failures to return (default: 10).

    Returns:
        Failed episodes sorted by validity (non-error finishes first), then turns and reward.
            Each dictionary contains:
            - rollout_id: Episode identifier
            - seed: Random seed
            - finish_reason: Termination cause
            - num_turns: Number of turns
            - cumulative_reward: Total reward
            - valid_action_count: Valid actions
            - action_check_count: Total action checks
            - observation_ablation: Ablation mode
            - transcript_path: Path to episode transcript
            - metrics_path: Path to metrics.json

    Example:
        >>> episodes = collect_evaluation_episodes(Path("outputs/eval"))
        >>> failures = representative_failures(episodes, limit=5)
        >>> for failure in failures:
        ...     print(f"{failure['rollout_id']}: {failure['finish_reason']}, "
        ...           f"{failure['num_turns']} turns, reward {failure['cumulative_reward']}")

    Note:
        Prioritizing non-error failures ensures that manual review focuses on semantic
        task failures (wrong actions, incorrect reasoning) rather than implementation
        bugs (environment crashes, model errors).
    """
    failures = [episode for episode in episodes if not episode["success"]]
    failures.sort(
        key=lambda episode: (
            episode["finish_reason"] not in {"model_error", "env_error", "error"},
            -episode["num_turns"],
            -episode["cumulative_reward"],
        )
    )
    return [
        {
            key: episode[key]
            for key in (
                "rollout_id",
                "seed",
                "finish_reason",
                "num_turns",
                "cumulative_reward",
                "valid_action_count",
                "action_check_count",
                "observation_ablation",
                "transcript_path",
                "metrics_path",
            )
        }
        for episode in failures[:limit]
    ]


def load_training_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load raw training rollout rows from JSONL files.

    Parses JSONL rollout dumps, skipping empty lines and validating JSON syntax.

    Args:
        paths: Paths to rollout JSONL files.

    Returns:
        List of row dictionaries with group_idx, traj_idx, turn_idx, and rollout data.
            Each row represents one environment turn with fields:
            - group_idx: Group identifier for GRPO
            - traj_idx: Trajectory index within group
            - turn_idx: Turn index within trajectory
            - score: Per-turn reward
            - output: Model response text
            - last_turn: Boolean terminal turn marker
            - traj_success: Boolean trajectory success flag
            - action_is_valid: Boolean action validation result

    Raises:
        ValueError: If any line contains invalid JSON, with file path and line number.

    Example:
        >>> paths = [Path("rollouts/1000.jsonl"), Path("rollouts/2000.jsonl")]
        >>> rows = load_training_rows(paths)
        >>> len(rows)
        512
        >>> rows[0]["group_idx"]
        'uuid-abc123'
        >>> rows[0]["turn_idx"]
        1

    Note:
        Non-dictionary JSON values are silently skipped to handle mixed-format logs.
        Empty lines are ignored.
    """
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _trajectory_turn_count(trajectory_rows: list[dict[str, Any]]) -> int:
    """Prefer a trajectory-level turn count while preserving old per-turn dumps.

    Current no-concat validation writes ``num_turns`` at the top level.  Older
    artifacts may contain ``__num_turns__`` or the same value inside
    ``reward_extra_info``.  A grouped per-turn rollout is allowed to carry a
    per-row value of one; in that case the row count remains authoritative.
    """
    explicit: list[int] = []
    for row in trajectory_rows:
        values = [row.get("num_turns"), row.get("__num_turns__")]
        reward_extra = row.get("reward_extra_info")
        if isinstance(reward_extra, dict):
            values.extend(
                [reward_extra.get("num_turns"), reward_extra.get("__num_turns__")]
            )
        for value in values:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                integer = int(value)
                if integer > 0 and float(value) == integer:
                    explicit.append(integer)
                    break
    if not explicit:
        return len(trajectory_rows)
    if len(set(explicit)) > 1:
        raise ValueError(f"conflicting trajectory num_turns values: {explicit}")
    candidate = explicit[0]
    if len(trajectory_rows) == 1 or candidate >= len(trajectory_rows):
        return candidate
    return len(trajectory_rows)


def analyze_training_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute group reward variance and trajectory statistics from rollout rows.

    Validates trajectory structure, deduplicates padding artifacts, and computes
    GRPO-relevant statistics including within-group reward variance.

    Args:
        rows: Rollout rows from load_training_rows.

    Returns:
        Summary with trajectory count, group variance, and zero-variance fraction:
            - rows: Total row count (including padding duplicates)
            - unique_rows: Deduplicated row count
            - padding_duplicates: Number of duplicate rows removed
            - trajectories: Total trajectory count
            - groups: Number of unique groups
            - success_rate: Fraction of successful trajectories
            - mean_turns_successful: Average turns for successful trajectories
            - mean_group_reward_variance: Average variance of rewards within groups
            - zero_variance_group_fraction: Fraction of groups with zero reward variance
            - answer_template_concentration: Max frequency of most common answer template
            - unique_answer_fraction: Ratio of unique answer templates to total outputs
            - invalid_action_fraction: Fraction of invalid actions
            - reward_turn_correlation: Correlation between turns and reward
            - reward_per_extra_turn_slope: Linear slope of reward vs turns
            - trajectory_records: List of per-trajectory statistics for export

    Raises:
        ValueError: If duplicate rows have conflicting data for the same (group, traj, turn).

    Example:
        >>> rows = load_training_rows([Path("rollouts/1000.jsonl")])
        >>> analysis = analyze_training_rows(rows)
        >>> analysis["mean_group_reward_variance"]
        0.15  # Higher variance enables better GRPO signal
        >>> analysis["zero_variance_group_fraction"]
        0.05  # Low is good - groups have diverse outcomes
        >>> analysis["padding_duplicates"]
        32  # Expected with distributed data parallelism

    Note:
        - High zero_variance_group_fraction (>0.3) indicates mode collapse or
          insufficient exploration within groups.
        - Padding duplicates are validated for consistency but excluded from statistics.
        - Rows missing (group_idx, traj_idx, turn_idx) are treated as ungrouped trajectories.
    """
    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    ungrouped: list[dict[str, Any]] = []
    duplicate_count = 0
    for row in rows:
        required = ("group_idx", "traj_idx", "turn_idx")
        if not all(field in row for field in required):
            ungrouped.append(row)
            continue
        key = (
            str(row["group_idx"]),
            int(row["traj_idx"]),
            int(row["turn_idx"]),
        )
        if key in unique:
            comparable = ("score", "output", "last_turn", "traj_success")
            if any(unique[key].get(field) != row.get(field) for field in comparable):
                raise ValueError(f"conflicting duplicate training row: {key}")
            duplicate_count += 1
        else:
            unique[key] = row

    trajectories: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for key, row in unique.items():
        trajectories[(key[0], key[1])].append(row)
    for index, row in enumerate(ungrouped):
        trajectories[(f"ungrouped-{index}", 0)].append(row)

    trajectory_records: list[dict[str, Any]] = []
    for (group, trajectory), trajectory_rows in trajectories.items():
        trajectory_rows.sort(key=lambda row: int(row.get("turn_idx", 0)))
        trajectory_records.append(
            {
                "group": group,
                "trajectory": trajectory,
                "reward": sum(float(row.get("score", 0.0)) for row in trajectory_rows),
                "turns": _trajectory_turn_count(trajectory_rows),
                "success": any(
                    bool(row.get("traj_success", False))
                    for row in trajectory_rows
                ),
            }
        )

    rewards_by_group: dict[str, list[float]] = defaultdict(list)
    for record in trajectory_records:
        rewards_by_group[record["group"]].append(record["reward"])
    comparable_variances = [
        float(np.var(rewards))
        for rewards in rewards_by_group.values()
        if len(rewards) >= 2
    ]
    successes = [record for record in trajectory_records if record["success"]]
    answer_concentration, unique_answer_fraction = _concentration(
        _answer_template(row.get("output", "")) for row in rows
    )
    invalid_values = [
        bool(row["action_is_valid"])
        for row in rows
        if "action_is_valid" in row
    ]
    correlation, slope = _correlation_and_slope(
        [float(record["turns"]) for record in trajectory_records],
        [float(record["reward"]) for record in trajectory_records],
    )

    return {
        "rows": len(rows),
        "unique_rows": len(unique) + len(ungrouped),
        "padding_duplicates": duplicate_count,
        "trajectories": len(trajectory_records),
        "groups": len(rewards_by_group),
        "success_rate": (
            len(successes) / len(trajectory_records)
            if trajectory_records
            else None
        ),
        "mean_turns_successful": (
            float(np.mean([record["turns"] for record in successes]))
            if successes
            else None
        ),
        "mean_group_reward_variance": (
            float(np.mean(comparable_variances))
            if comparable_variances
            else None
        ),
        "zero_variance_group_fraction": (
            float(np.mean(np.asarray(comparable_variances) == 0.0))
            if comparable_variances
            else None
        ),
        "answer_template_concentration": answer_concentration,
        "unique_answer_fraction": unique_answer_fraction,
        "invalid_action_fraction": (
            1.0 - float(np.mean(invalid_values)) if invalid_values else None
        ),
        "reward_turn_correlation": correlation,
        "reward_per_extra_turn_slope": slope,
        "trajectory_records": trajectory_records,
    }


def _latest_jsonl(root: Path) -> Path | None:
    paths = list((root / "validation").glob("*.jsonl"))
    if not paths:
        paths = list((root / "rollouts").glob("*.jsonl"))
    if not paths:
        return None

    def order(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return -1, path.name

    return max(paths, key=order)


def build_result_row(root: Path) -> dict[str, Any]:
    """Build a result table row from training or evaluation artifacts.

    Aggregates metrics from multiple sources (manifest, episodes, GPU samples, parity checks)
    to produce a single row for the main results table with completeness validation.

    Args:
        root: Run directory containing manifest, metrics, and GPU samples.
            Expected structure:
            - manifest.json: Run configuration and provenance
            - metrics.json files: Evaluation episode results (recursive search)
            - rollouts/*.jsonl or validation/*.jsonl: Training rollouts
            - gpu_metrics/gpu_summary.json: Resource usage
            - parity.json: Rollout/train log-probability checks
            - train_command.sh / eval_command.sh: Reproducibility artifacts
            - resolved_config.yaml / resolved_config.txt: Full configuration

    Returns:
        Result row dictionary with completeness status and evidence paths:
            - Method: Experiment method name from manifest
            - Visual Success: Success rate (evaluation or latest training rollout)
            - Peak VRAM: Peak GPU memory usage in MiB
            - GPU·h: Total GPU-hours consumed
            - Mean Turns: Average turns for successful episodes/trajectories
            - Ratio P95: 95th percentile rollout/train log-prob ratio
            - Status: "complete", "incomplete-artifacts", "failed", or "not-run"
            - Environment: Environment name
            - Seed: Random seed or seed range start
            - Commit: Git commit hash
            - Evidence: Semicolon-separated list of source file paths

    Example:
        >>> row = build_result_row(Path("outputs/grpo_run_001"))
        >>> row["Status"]
        'complete'
        >>> row["Visual Success"]
        0.73
        >>> row["Peak VRAM"]
        45120
        >>> row["GPU·h"]
        12.5

    Note:
        Status determination:
        - "failed": GPU exit code != 0, evaluation errors, or parity gate failures
        - "complete": All artifacts present, counts match expectations, parity passed
        - "incomplete-artifacts": Some artifacts exist but validation incomplete
        - "not-run": No artifacts found

        Provenance completeness requires clean git state and preserved command/config files.
    """
    manifest = _load_json(root / "manifest.json") or {}
    episodes = collect_evaluation_episodes(root)
    latest = None
    evidence: list[str] = [
        str(path)
        for path in (
            root / "manifest.json",
            root / "train_command.sh",
            root / "eval_command.sh",
            root / "resolved_config.yaml",
            root / "resolved_config.txt",
        )
        if path.exists()
    ]
    behavior_complete = False
    evaluation_failed = False
    visual_success = None
    mean_turns = None
    if episodes:
        evaluation = analyze_evaluation_episodes(episodes)
        evaluation_failed = any(
            episode["finish_reason"] not in {"done", "max_turns"}
            for episode in episodes
        )
        visual_success = evaluation["success_rate"]
        mean_turns = evaluation["mean_turns_successful"]
        evidence.extend(episode["metrics_path"] for episode in episodes[:1])
        expected_episodes = manifest.get("n_envs")
        behavior_complete = (
            isinstance(expected_episodes, int)
            and expected_episodes > 0
            and len(episodes) == expected_episodes
            and len({episode["seed"] for episode in episodes})
            == expected_episodes
        )
    else:
        latest = _latest_jsonl(root)
        if latest:
            training = analyze_training_rows(load_training_rows([latest]))
            visual_success = training["success_rate"]
            mean_turns = training["mean_turns_successful"]
            evidence.append(str(latest))
            expected_step = int(manifest.get("total_steps", 0))
            actual_step = int(latest.stem) if latest.stem.isdigit() else 0
            expected_episodes = int(manifest.get("validation_n_envs", 0))
            actual_episodes = training.get("trajectories", 0)
            behavior_complete = (
                expected_step > 0
                and actual_step == expected_step
                and expected_episodes > 0
                and actual_episodes == expected_episodes
            )

    gpu = _load_json(root / "gpu_metrics" / "gpu_summary.json") or {}
    parity = _load_json(root / "parity.json") or {}
    if gpu:
        evidence.append(str(root / "gpu_metrics" / "gpu_summary.json"))
    if parity:
        evidence.append(str(root / "parity.json"))
    parity_metrics = parity.get("metrics") if isinstance(parity.get("metrics"), dict) else {}
    gpu_complete = (
        gpu.get("return_code") == 0
        and int(gpu.get("sample_count", 0) or 0) > 0
        and gpu.get("peak_vram_mib") is not None
        and gpu.get("gpu_hours") is not None
        and not gpu.get("sampling_errors")
        and (
            gpu.get("expected_device_count") is None
            or gpu.get("expected_device_count") == gpu.get("gpu_count")
        )
    )
    is_training = "advantage_estimator" in manifest
    parity_complete = (
        not is_training
        or (
            parity.get("gate_enabled") is True
            and parity.get("gate_passed") is True
        )
    )
    parity_attempts = parity.get("attempts")
    parity_failed_once = (
        isinstance(parity_attempts, list)
        and any(
            isinstance(attempt, dict) and attempt.get("gate_passed") is False
            for attempt in parity_attempts
        )
    )

    def nonempty_file(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    provenance_complete = bool(
        manifest.get("commit")
        and manifest.get("verl_commit")
        and manifest.get("git_dirty") is False
        and (
            (
                nonempty_file(root / "train_command.sh")
                and nonempty_file(root / "resolved_config.yaml")
            )
            if is_training
            else (
                nonempty_file(root / "eval_command.sh")
                and nonempty_file(root / "resolved_config.txt")
            )
        )
    )
    failed = (
        (gpu and gpu.get("return_code") not in {None, 0})
        or evaluation_failed
        or (
            is_training
            and (
                parity.get("gate_passed") is False
                or parity_failed_once
            )
        )
    )
    if failed:
        status = "failed"
    elif behavior_complete and gpu_complete and parity_complete and provenance_complete:
        status = "complete"
    elif evidence or episodes or latest or gpu or parity:
        status = "incomplete-artifacts"
    else:
        status = "not-run"

    return {
        "Method": manifest.get("method", "unknown"),
        "Visual Success": visual_success,
        "Peak VRAM": gpu.get("peak_vram_mib"),
        "GPU·h": gpu.get("gpu_hours"),
        "Mean Turns": mean_turns,
        "Ratio P95": parity_metrics.get("ratio_p95"),
        "Status": status,
        "Environment": manifest.get("environment"),
        "Seed": manifest.get("seed", manifest.get("seed_start")),
        "Commit": manifest.get("commit"),
        "Evidence": ";".join(evidence),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    """Analyze rollouts and evaluation episodes, writing summaries and result rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dump", type=Path, action="append", default=[])
    parser.add_argument("--training-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (args.eval_dump or args.training_jsonl or args.run):
        parser.error("provide --eval-dump, --training-jsonl, or --run")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    all_episodes: list[dict[str, Any]] = []
    for root in args.eval_dump:
        all_episodes.extend(collect_evaluation_episodes(root))
    if all_episodes:
        report["evaluation"] = analyze_evaluation_episodes(all_episodes)
        _write_csv(
            args.output_dir / "evaluation_episodes.csv",
            all_episodes,
            [
                "rollout_id",
                "seed",
                "success",
                "num_turns",
                "cumulative_reward",
                "finish_reason",
                "environment",
                "tag_id",
                "observation_ablation",
                "valid_action_count",
                "action_check_count",
                "metrics_path",
                "transcript_path",
            ],
        )
        _write_jsonl(
            args.output_dir / "failure_cases.jsonl",
            representative_failures(all_episodes),
        )

    if args.training_jsonl:
        training_rows = load_training_rows(args.training_jsonl)
        training_report = analyze_training_rows(training_rows)
        trajectory_records = training_report.pop("trajectory_records")
        report["training"] = training_report
        _write_csv(
            args.output_dir / "training_trajectories.csv",
            trajectory_records,
            ["group", "trajectory", "reward", "turns", "success"],
        )

    if args.run:
        result_rows = [build_result_row(root) for root in args.run]
        _write_csv(
            args.output_dir / "main_results.csv",
            result_rows,
            RESULT_COLUMNS,
        )
        report["result_rows"] = result_rows

    (args.output_dir / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
