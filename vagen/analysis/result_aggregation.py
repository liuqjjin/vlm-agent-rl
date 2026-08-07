"""Aggregate experimental results across seeds, methods, and environments.

This module provides tools for aggregating results from multiple runs to support
the P1 requirement of comparing methods across 2-3 seeds and multiple environments.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def aggregate_across_seeds(
    run_dirs: list[Path],
    method_name: str | None = None,
    *,
    require_evaluation_role: str | None = None,
    environment: str | None = None,
    expected_seeds: set[int] | None = None,
) -> dict[str, Any]:
    """Aggregate results from multiple seeds of the same method.

    Args:
        run_dirs: List of run directories for the same method with different seeds
        method_name: Optional method name override (extracted from manifest if None)

    Returns:
        Dictionary with aggregated metrics including means, std, and per-seed results

    Example:
        >>> from pathlib import Path
        >>> runs = [
        ...     Path("exps/concat_grpo_seed0"),
        ...     Path("exps/concat_grpo_seed1"),
        ...     Path("exps/concat_grpo_seed2"),
        ... ]
        >>> result = aggregate_across_seeds(runs)
        >>> print(f"Success rate: {result['success_rate']['mean']:.3f} ± {result['success_rate']['std']:.3f}")
    """
    from vagen.analysis.statistical_analysis import load_method_results

    if not run_dirs:
        raise ValueError("run_dirs cannot be empty")

    results = [
        load_method_results(
            run_dir,
            require_evaluation_role=require_evaluation_role,
            environment=environment,
        )
        for run_dir in run_dirs
    ]

    # Extract method name
    inferred_method = method_name is None
    if inferred_method:
        method_name = results[0].get("method", "unknown")

    # Check consistency
    methods = {r.get("method") for r in results}
    if inferred_method and len(methods) > 1:
        raise ValueError(f"Multiple methods found: {methods}. Use method_name to override.")

    environments = {r.get("environment") for r in results}
    if len(environments) > 1:
        raise ValueError(f"Multiple environments found: {environments}")

    seeds = [result.get("seed") for result in results]
    if expected_seeds is not None:
        actual_seeds = {seed for seed in seeds if isinstance(seed, int)}
        if len(seeds) != len(actual_seeds):
            raise ValueError("seed aggregation contains duplicate or non-integer seeds")
        if actual_seeds != expected_seeds:
            raise ValueError(
                f"Expected seeds {sorted(expected_seeds)}, found {sorted(actual_seeds)}"
            )

    # Aggregate success rates
    success_rates = [r["success_rate"] for r in results if r["success_rate"] is not None]
    total_successes = sum(r["successful_trajectories"] for r in results)
    total_trials = sum(r["trajectories"] for r in results)

    # Aggregate mean turns (successful episodes only)
    mean_turns_values = [r["mean_turns"] for r in results if r["mean_turns"] is not None]
    all_turn_values = [turn for r in results for turn in r.get("turn_values", [])]

    # Aggregate GPU metrics
    gpu_hours_values = [r["gpu_hours"] for r in results if r["gpu_hours"] is not None]
    peak_vram_values = [r["peak_vram_mib"] for r in results if r["peak_vram_mib"] is not None]

    return {
        "method": method_name,
        "environment": results[0].get("environment"),
        "n_seeds": len(results),
        "seeds": seeds,
        "evaluation_role": require_evaluation_role,
        "success_rate": {
            "mean": float(np.mean(success_rates)) if success_rates else None,
            "std": float(np.std(success_rates, ddof=1)) if len(success_rates) > 1 else 0.0,
            "per_seed": success_rates,
            "pooled": total_successes / total_trials if total_trials > 0 else None,
            "total_successes": total_successes,
            "total_trials": total_trials,
        },
        "mean_turns": {
            "mean": float(np.mean(mean_turns_values)) if mean_turns_values else None,
            "std": float(np.std(mean_turns_values, ddof=1)) if len(mean_turns_values) > 1 else 0.0,
            "per_seed": mean_turns_values,
            "pooled_all_episodes": float(np.mean(all_turn_values)) if all_turn_values else None,
        },
        "gpu_hours": {
            "mean": float(np.mean(gpu_hours_values)) if gpu_hours_values else None,
            "std": float(np.std(gpu_hours_values, ddof=1)) if len(gpu_hours_values) > 1 else 0.0,
            "total": float(np.sum(gpu_hours_values)) if gpu_hours_values else None,
            "per_seed": gpu_hours_values,
        },
        "peak_vram_mib": {
            "mean": float(np.mean(peak_vram_values)) if peak_vram_values else None,
            "max": float(np.max(peak_vram_values)) if peak_vram_values else None,
            "per_seed": peak_vram_values,
        },
    }


def compare_across_environments(
    method_runs_by_env: dict[str, list[Path]],
    method_name: str | None = None,
) -> dict[str, Any]:
    """Compare same method across different environments (Sokoban vs Navigation).

    Args:
        method_runs_by_env: Dictionary mapping environment names to lists of run directories
        method_name: Optional method name

    Returns:
        Dictionary with per-environment aggregated results and comparisons

    Example:
        >>> runs = {
        ...     "sokoban": [Path("exps/grpo_sokoban_s0"), Path("exps/grpo_sokoban_s1")],
        ...     "navigation": [Path("exps/grpo_nav_s0"), Path("exps/grpo_nav_s1")],
        ... }
        >>> result = compare_across_environments(runs, "concat_grpo")
        >>> print(result["comparison"]["success_easier_on"])
    """
    if len(method_runs_by_env) < 2:
        raise ValueError("Need at least 2 environments to compare")

    # Aggregate each environment
    env_results = {}
    for env_name, run_dirs in method_runs_by_env.items():
        env_results[env_name] = aggregate_across_seeds(run_dirs, method_name)

    # Compare environments
    env_names = list(env_results.keys())
    success_rates = {env: env_results[env]["success_rate"]["pooled"] for env in env_names}

    # Determine which environment is easier
    easier_env = max(success_rates.keys(), key=lambda e: success_rates.get(e) or 0.0)
    harder_env = min(success_rates.keys(), key=lambda e: success_rates.get(e) or 1.0)

    return {
        "method": method_name or env_results[env_names[0]]["method"],
        "environments": env_names,
        "per_environment": env_results,
        "comparison": {
            "success_easier_on": easier_env,
            "success_harder_on": harder_env,
            "success_rate_gap": (success_rates.get(easier_env) or 0.0) - (success_rates.get(harder_env) or 0.0),
            "mean_turns_by_env": {
                env: env_results[env]["mean_turns"]["pooled_all_episodes"]
                for env in env_names
            },
        },
    }


def aggregate_experiment_matrix(
    run_dirs: list[Path],
    group_by: str = "method",
    *,
    require_evaluation_role: str | None = None,
    expected_seeds: set[int] | None = None,
) -> dict[str, Any]:
    """Aggregate results from a full experiment matrix.

    Args:
        run_dirs: List of all run directories to aggregate
        group_by: How to group runs ("method", "environment", or "seed")

    Returns:
        Structured aggregation of the full experiment matrix

    Example:
        >>> all_runs = list(Path("exps/vlm_agent_rl").glob("*/"))
        >>> matrix = aggregate_experiment_matrix(all_runs, group_by="method")
        >>> for method, data in matrix["groups"].items():
        ...     print(f"{method}: {data['success_rate']['mean']:.2f}")
    """
    from vagen.analysis.statistical_analysis import load_method_results

    if not run_dirs:
        return {"groups": {}, "n_runs": 0}

    # Load all results
    all_results = []
    for run_dir in run_dirs:
        try:
            result = load_method_results(
                run_dir, require_evaluation_role=require_evaluation_role
            )
            result["run_dir"] = str(run_dir)
            all_results.append(result)
        except Exception:
            # Skip failed runs
            continue

    # Group by specified key
    groups = defaultdict(list)
    for result in all_results:
        key = result.get(group_by, "unknown")
        groups[str(key)].append(Path(result["run_dir"]))

    # Aggregate each group
    aggregated = {}
    for group_key, group_runs in groups.items():
        try:
            aggregated[group_key] = aggregate_across_seeds(
                group_runs,
                require_evaluation_role=require_evaluation_role,
                expected_seeds=expected_seeds,
            )
        except ValueError as e:
            # Handle inconsistent groups
            aggregated[group_key] = {"error": str(e), "run_dirs": [str(r) for r in group_runs]}

    return {
        "group_by": group_by,
        "evaluation_role": require_evaluation_role,
        "n_runs": len(all_results),
        "n_groups": len(aggregated),
        "groups": aggregated,
    }


def compute_seed_stability_metrics(
    success_rates: list[float],
    mean_turns: list[float],
) -> dict[str, Any]:
    """Compute stability metrics across seeds.

    Args:
        success_rates: Success rates from different seeds
        mean_turns: Mean turn counts from different seeds

    Returns:
        Dictionary with coefficient of variation and other stability metrics

    Example:
        >>> success_rates = [0.45, 0.52, 0.48]
        >>> mean_turns = [6.5, 6.2, 6.8]
        >>> stability = compute_seed_stability_metrics(success_rates, mean_turns)
        >>> print(f"Success CV: {stability['success_rate_cv']:.3f}")
    """
    if len(success_rates) < 2:
        return {
            "success_rate_cv": None,
            "mean_turns_cv": None,
            "success_rate_range": None,
            "interpretation": "insufficient_seeds",
        }

    # Coefficient of variation (std / mean)
    success_mean = np.mean(success_rates)
    success_std = np.std(success_rates, ddof=1)
    success_cv = float(success_std / success_mean) if success_mean > 0 else float("inf")

    turns_mean = np.mean(mean_turns)
    turns_std = np.std(mean_turns, ddof=1)
    turns_cv = float(turns_std / turns_mean) if turns_mean > 0 else float("inf")

    # Range
    success_range = float(np.max(success_rates) - np.min(success_rates))

    # Interpretation
    if success_cv < 0.1:
        interpretation = "highly_stable"
    elif success_cv < 0.2:
        interpretation = "stable"
    elif success_cv < 0.5:
        interpretation = "moderate_variance"
    else:
        interpretation = "high_variance"

    return {
        "success_rate_cv": success_cv,
        "mean_turns_cv": turns_cv,
        "success_rate_range": success_range,
        "success_rate_min": float(np.min(success_rates)),
        "success_rate_max": float(np.max(success_rates)),
        "interpretation": interpretation,
    }


def generate_comparison_table(
    aggregated_methods: dict[str, dict[str, Any]],
    output_path: Path | None = None,
) -> str:
    """Generate a markdown comparison table from aggregated results.

    Args:
        aggregated_methods: Dictionary mapping method names to aggregated results
        output_path: Optional path to save markdown file

    Returns:
        Markdown table as string

    Example:
        >>> methods = {
        ...     "concat_grpo": aggregate_across_seeds(concat_runs),
        ...     "no_concat_grpo": aggregate_across_seeds(no_concat_runs),
        ... }
        >>> table = generate_comparison_table(methods)
        >>> print(table)
    """
    lines = [
        "# Method Comparison",
        "",
        "| Method | Success Rate | Mean Turns | Peak VRAM (MiB) | GPU Hours | Seeds |",
        "|--------|--------------|------------|-----------------|-----------|-------|",
    ]

    for method_name, data in sorted(aggregated_methods.items()):
        success = data["success_rate"]
        turns = data["mean_turns"]
        vram = data["peak_vram_mib"]
        gpu = data["gpu_hours"]

        success_str = (
            f"{success['mean']:.3f} ± {success['std']:.3f}"
            if success["mean"] is not None
            else "N/A"
        )
        turns_str = (
            f"{turns['mean']:.1f} ± {turns['std']:.1f}"
            if turns["mean"] is not None
            else "N/A"
        )
        vram_str = f"{vram['max']:.0f}" if vram["max"] is not None else "N/A"
        gpu_str = f"{gpu['total']:.1f}" if gpu["total"] is not None else "N/A"

        lines.append(
            f"| {method_name} | {success_str} | {turns_str} | {vram_str} | {gpu_str} | {data['n_seeds']} |"
        )

    markdown = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)

    return markdown


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--group-by", default="method", choices=["method", "environment", "seed"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix = aggregate_experiment_matrix(args.run, group_by=args.group_by)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, indent=2, ensure_ascii=False))

    print(f"Aggregated {matrix['n_runs']} runs into {matrix['n_groups']} groups")
    print(f"Results saved to {args.output}")
