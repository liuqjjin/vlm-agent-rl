"""Compare token, turn, and trajectory policy objective aggregation modes.

This module implements P1 analysis for comparing the three policy objective
aggregation modes: token-level, turn-level, and trajectory-level weighting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def extract_objective_mode(manifest: dict[str, Any]) -> str:
    """Extract the policy objective aggregation mode from manifest.

    Args:
        manifest: Run manifest dictionary

    Returns:
        One of "token", "turn", "trajectory", or "unknown"
    """
    loss_weighting = manifest.get("loss_weighting")
    if str(loss_weighting).lower() in {"token", "turn", "trajectory"}:
        return str(loss_weighting).lower()

    # Check for policy_objective_aggregation or similar keys
    policy_config = manifest.get("policy_objective", {})
    if isinstance(policy_config, dict):
        mode = policy_config.get("aggregation", policy_config.get("mode"))
        if mode:
            return str(mode).lower()

    # Fallback: check method name
    method = manifest.get("method", "")
    if "token" in method.lower():
        return "token"
    elif "turn" in method.lower():
        return "turn"
    elif "trajectory" in method.lower() or "episode" in method.lower():
        return "trajectory"

    return "unknown"


def compare_objective_modes(
    run_dirs_by_mode: dict[str, list[Path]],
    environment: str | None = None,
    *,
    require_evaluation_role: str | None = "final_test",
    expected_seeds: set[int] | None = None,
) -> dict[str, Any]:
    """Compare token, turn, and trajectory objective aggregation modes.

    Args:
        run_dirs_by_mode: Dictionary mapping objective mode to run directories
        environment: Optional environment filter

    Returns:
        Comprehensive comparison of the three objective modes

    Example:
        >>> runs = {
        ...     "token": [Path("exps/token_s0"), Path("exps/token_s1")],
        ...     "turn": [Path("exps/turn_s0"), Path("exps/turn_s1")],
        ...     "trajectory": [Path("exps/traj_s0"), Path("exps/traj_s1")],
        ... }
        >>> comparison = compare_objective_modes(runs)
        >>> print(f"Best mode: {comparison['ranking']['by_success'][0]['mode']}")
    """
    from vagen.analysis.result_aggregation import aggregate_across_seeds
    from vagen.analysis.statistical_analysis import compare_success_rates, compare_mean_turns

    if len(run_dirs_by_mode) < 2:
        raise ValueError("Need at least 2 objective modes to compare")

    # Aggregate each mode
    mode_results = {}
    for mode, run_dirs in run_dirs_by_mode.items():
        mode_results[mode] = aggregate_across_seeds(
            run_dirs,
            method_name=mode,
            require_evaluation_role=require_evaluation_role,
            environment=environment,
            expected_seeds=expected_seeds,
        )

    # Extract metrics for ranking
    modes = list(mode_results.keys())
    success_rates = {
        mode: mode_results[mode]["success_rate"]["pooled"]
        for mode in modes
        if mode_results[mode]["success_rate"]["pooled"] is not None
    }
    mean_turns = {
        mode: mode_results[mode]["mean_turns"]["pooled_all_episodes"]
        for mode in modes
        if mode_results[mode]["mean_turns"]["pooled_all_episodes"] is not None
    }
    gpu_hours = {
        mode: mode_results[mode]["gpu_hours"]["total"]
        for mode in modes
        if mode_results[mode]["gpu_hours"]["total"] is not None
    }

    # Compute resource efficiency (success per GPU hour). This is not sample
    # efficiency because no interaction/token denominator is available.
    compute_efficiency = {}
    for mode in modes:
        if mode in success_rates and mode in gpu_hours:
            if gpu_hours[mode] > 0:
                compute_efficiency[mode] = success_rates[mode] / gpu_hours[mode]

    # Rank by different criteria
    ranking_by_success = sorted(
        success_rates.items(), key=lambda x: x[1], reverse=True
    )
    ranking_by_efficiency = sorted(
        compute_efficiency.items(), key=lambda x: x[1], reverse=True
    )
    ranking_by_turns = sorted(
        mean_turns.items(), key=lambda x: x[1]  # Lower is better
    )

    # Pairwise statistical comparisons
    pairwise_comparisons = {}
    for i, mode1 in enumerate(modes):
        for mode2 in modes[i + 1:]:
            key = f"{mode1}_vs_{mode2}"

            # Success rate comparison
            result1 = mode_results[mode1]
            result2 = mode_results[mode2]

            success_comp = compare_success_rates(
                mode1,
                result1["success_rate"]["total_successes"],
                result1["success_rate"]["total_trials"],
                mode2,
                result2["success_rate"]["total_successes"],
                result2["success_rate"]["total_trials"],
            )

            # Turn comparison (if both have data)
            turns1 = [
                turn
                for run in run_dirs_by_mode[mode1]
                for turn in _load_turn_values(
                    run, require_evaluation_role, environment
                )
            ]
            turns2 = [
                turn
                for run in run_dirs_by_mode[mode2]
                for turn in _load_turn_values(
                    run, require_evaluation_role, environment
                )
            ]

            turns_comp = None
            if turns1 and turns2:
                turns_comp = compare_mean_turns(mode1, turns1, mode2, turns2)

            pairwise_comparisons[key] = {
                "success_rate": success_comp,
                "mean_turns": turns_comp,
            }

    return {
        "environment": environment,
        "objective_modes": modes,
        "per_mode": mode_results,
        "metrics": {
            "success_rates": success_rates,
            "mean_turns": mean_turns,
            "gpu_hours": gpu_hours,
            "compute_efficiency": compute_efficiency,
        },
        "ranking": {
            "by_success": [
                {"mode": mode, "success_rate": rate}
                for mode, rate in ranking_by_success
            ],
            "by_efficiency": [
                {"mode": mode, "efficiency": eff}
                for mode, eff in ranking_by_efficiency
            ],
            "by_turns": [
                {"mode": mode, "mean_turns": turns}
                for mode, turns in ranking_by_turns
            ],
        },
        "pairwise_comparisons": pairwise_comparisons,
        "interpretation": _interpret_objective_comparison(
            ranking_by_success, ranking_by_efficiency, pairwise_comparisons
        ),
    }


def _load_turn_values(
    run_dir: Path,
    require_evaluation_role: str | None,
    environment: str | None,
) -> list[float]:
    """Load turn values from a run directory."""
    from vagen.analysis.statistical_analysis import load_method_results

    try:
        result = load_method_results(
            run_dir,
            require_evaluation_role=require_evaluation_role,
            environment=environment,
        )
        return result.get("turn_values", [])
    except Exception:
        return []


def _interpret_objective_comparison(
    ranking_by_success: list[tuple[str, float]],
    ranking_by_efficiency: list[tuple[str, float]],
    pairwise_comparisons: dict[str, Any],
) -> str:
    """Generate human-readable interpretation of objective mode comparison."""
    if not ranking_by_success:
        return "No valid comparisons available"

    best_success = ranking_by_success[0][0]
    best_efficiency = ranking_by_efficiency[0][0] if ranking_by_efficiency else None

    parts = []
    parts.append(f"Best success rate: {best_success}")

    if best_efficiency:
        if best_efficiency == best_success:
            parts.append(f"{best_success} also achieves highest compute efficiency")
        else:
            parts.append(f"Most compute-efficient: {best_efficiency}")

    # Check for significant differences
    significant_pairs = []
    for pair_key, comparison in pairwise_comparisons.items():
        success_comp = comparison["success_rate"]
        if success_comp["significance_test"]["significant_at_0.05"]:
            mode1, mode2 = pair_key.split("_vs_")
            winner = mode1 if success_comp["difference"]["absolute"] > 0 else mode2
            significant_pairs.append(f"{winner} significantly outperforms")

    if significant_pairs:
        parts.extend(significant_pairs[:2])  # Limit to top 2
    else:
        parts.append("No statistically significant differences detected")

    return "; ".join(parts)


def analyze_objective_gradient_stability(
    training_log_paths: dict[str, Path],
) -> dict[str, Any]:
    """Analyze gradient stability across objective modes from training logs.

    Args:
        training_log_paths: Dictionary mapping objective mode to training log path

    Returns:
        Gradient norm statistics and stability metrics for each mode

    Example:
        >>> logs = {
        ...     "token": Path("exps/token_s0/training.log"),
        ...     "turn": Path("exps/turn_s0/training.log"),
        ...     "trajectory": Path("exps/traj_s0/training.log"),
        ... }
        >>> stability = analyze_objective_gradient_stability(logs)
        >>> print(f"Most stable: {stability['most_stable_mode']}")
    """
    gradient_stats = {}

    for mode, log_path in training_log_paths.items():
        if not log_path.exists():
            continue

        # Parse gradient norms from log
        gradient_norms = []
        try:
            with log_path.open() as f:
                for line in f:
                    # Look for gradient norm patterns (adjust based on actual log format)
                    if "actor_grad_norm" in line or "policy_grad_norm" in line:
                        try:
                            # Extract numeric value (simplified parsing)
                            parts = line.split(":")
                            for part in parts:
                                if "grad_norm" in part:
                                    idx = parts.index(part)
                                    if idx + 1 < len(parts):
                                        value = float(parts[idx + 1].split()[0].strip(","))
                                        gradient_norms.append(value)
                        except (ValueError, IndexError):
                            continue
        except Exception:
            pass

        if gradient_norms:
            gradient_stats[mode] = {
                "mean": float(np.mean(gradient_norms)),
                "std": float(np.std(gradient_norms)),
                "median": float(np.median(gradient_norms)),
                "max": float(np.max(gradient_norms)),
                "cv": float(np.std(gradient_norms) / np.mean(gradient_norms)),
                "n_samples": len(gradient_norms),
            }

    # Determine most stable (lowest CV)
    most_stable = None
    if gradient_stats:
        most_stable = min(gradient_stats.keys(), key=lambda m: gradient_stats[m]["cv"])

    return {
        "per_mode": gradient_stats,
        "most_stable_mode": most_stable,
    }


def generate_objective_comparison_report(
    comparison: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate a comprehensive markdown report for objective mode comparison.

    Args:
        comparison: Output from compare_objective_modes
        output_path: Path to save markdown report
    """
    lines = [
        "# Policy Objective Aggregation Comparison",
        "",
        f"**Environment:** {comparison.get('environment', 'N/A')}",
        "",
        "## Summary",
        "",
        comparison["interpretation"],
        "",
        "## Metrics by Objective Mode",
        "",
        "| Mode | Success Rate | Mean Turns | GPU Hours | Efficiency |",
        "|------|--------------|------------|-----------|------------|",
    ]

    metrics = comparison["metrics"]
    for mode in comparison["objective_modes"]:
        success = metrics["success_rates"].get(mode, 0.0)
        turns = metrics["mean_turns"].get(mode, 0.0)
        gpu = metrics["gpu_hours"].get(mode, 0.0)
        eff = metrics["compute_efficiency"].get(mode, 0.0)

        lines.append(
            f"| {mode} | {success:.3f} | {turns:.1f} | {gpu:.1f} | {eff:.4f} |"
        )

    lines.extend([
        "",
        "## Ranking",
        "",
        "### By Success Rate",
        "",
    ])

    for rank in comparison["ranking"]["by_success"]:
        lines.append(f"- **{rank['mode']}**: {rank['success_rate']:.3f}")

    lines.extend([
        "",
        "### By Compute Efficiency",
        "",
    ])

    for rank in comparison["ranking"]["by_efficiency"]:
        lines.append(f"- **{rank['mode']}**: {rank['efficiency']:.4f} success/GPU·h")

    lines.extend([
        "",
        "## Pairwise Comparisons",
        "",
    ])

    for pair_key, comp in comparison["pairwise_comparisons"].items():
        mode1, mode2 = pair_key.split("_vs_")
        success_comp = comp["success_rate"]
        sig = success_comp["significance_test"]

        lines.append(f"### {mode1} vs {mode2}")
        lines.append("")
        lines.append(f"- Success rate difference: {success_comp['difference']['absolute']:.3f}")
        lines.append(f"- p-value: {sig['p_value']:.4f}")
        lines.append(f"- Significant: {'Yes' if sig['significant_at_0.05'] else 'No'}")
        lines.append("")

    markdown = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser("compare", help="Compare objective modes")
    compare_parser.add_argument("--token-runs", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--turn-runs", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--trajectory-runs", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--environment", type=str)
    compare_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "compare":
        runs = {
            "token": args.token_runs,
            "turn": args.turn_runs,
            "trajectory": args.trajectory_runs,
        }
        comparison = compare_objective_modes(runs, args.environment)

        # Save JSON
        json_path = args.output.with_suffix(".json")
        json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))

        # Generate markdown report
        md_path = args.output.with_suffix(".md")
        generate_objective_comparison_report(comparison, md_path)

        print(f"Comparison saved to {json_path} and {md_path}")
        print(f"Interpretation: {comparison['interpretation']}")
