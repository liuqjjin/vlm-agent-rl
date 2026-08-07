"""Statistical analysis tools for comparing RL training methods.

Provides confidence intervals, effect sizes, significance tests, and sample
efficiency metrics for experimental result interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def wilson_score_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score confidence interval for binomial proportion.

    More accurate than normal approximation for small samples or extreme proportions.

    Args:
        successes: Number of successful trials
        trials: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) tuple

    Example:
        >>> wilson_score_interval(45, 100, 0.95)
        (0.352, 0.549)
    """
    if trials == 0:
        return 0.0, 0.0
    if successes < 0 or successes > trials:
        raise ValueError(f"successes ({successes}) must be in [0, {trials}]")

    p = successes / trials
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    z2 = z * z

    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials) / denominator

    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_mean_ci(
    values: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for mean.

    Args:
        values: Sample values
        confidence: Confidence level
        n_bootstrap: Number of bootstrap resamples
        random_seed: Random seed for reproducibility

    Returns:
        (mean, lower_bound, upper_bound) tuple
    """
    if not values:
        return 0.0, 0.0, 0.0

    values_array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values_array))

    if len(values) == 1:
        return mean, mean, mean

    rng = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(values_array, size=len(values_array), replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return mean, lower, upper


def cohens_d(
    group1: list[float], group2: list[float], pooled: bool = True
) -> float:
    """Cohen's d effect size for comparing two groups.

    Args:
        group1: First group values
        group2: Second group values
        pooled: Use pooled standard deviation (default True)

    Returns:
        Effect size. Interpretation: 0.2=small, 0.5=medium, 0.8=large

    Example:
        >>> cohens_d([6.5, 6.8, 6.3], [7.2, 7.5, 7.1])  # method A vs B mean turns
        -0.87  # negative means group2 has higher values (more turns)
    """
    if not group1 or not group2:
        return 0.0

    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)

    mean1, mean2 = np.mean(g1), np.mean(g2)

    if pooled:
        n1, n2 = len(g1), len(g2)
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        denominator = pooled_std
    else:
        denominator = np.std(g1, ddof=1)

    if denominator == 0:
        return 0.0

    return float((mean1 - mean2) / denominator)


def mann_whitney_u_test(
    group1: list[float], group2: list[float]
) -> tuple[float, float]:
    """Mann-Whitney U test for comparing two independent groups (non-parametric).

    More robust than t-test when distributions are not normal or sample sizes are small.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        (u_statistic, p_value) tuple. p < 0.05 suggests groups differ significantly.
    """
    if len(group1) < 1 or len(group2) < 1:
        return 0.0, 1.0

    result = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def fisher_exact_test(
    success1: int, trials1: int, success2: int, trials2: int
) -> tuple[float, float]:
    """Fisher's exact test for comparing two binomial proportions.

    Exact test for 2×2 contingency table. Use when sample sizes are small.

    Args:
        success1: Successes in group 1
        trials1: Total trials in group 1
        success2: Successes in group 2
        trials2: Total trials in group 2

    Returns:
        (odds_ratio, p_value) tuple
    """
    table = [
        [success1, trials1 - success1],
        [success2, trials2 - success2],
    ]
    odds_ratio, p_value = stats.fisher_exact(table)
    return float(odds_ratio), float(p_value)


def compare_success_rates(
    method1_name: str,
    method1_successes: int,
    method1_trials: int,
    method2_name: str,
    method2_successes: int,
    method2_trials: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compare success rates between two methods with statistical tests.

    Args:
        method1_name: Name of first method
        method1_successes: Number of successful episodes for method 1
        method1_trials: Total episodes for method 1
        method2_name: Name of second method
        method2_successes: Number of successful episodes for method 2
        method2_trials: Total episodes for method 2
        confidence: Confidence level for intervals

    Returns:
        Dictionary with success rates, CIs, difference, and significance test
    """
    rate1 = method1_successes / method1_trials if method1_trials > 0 else 0.0
    rate2 = method2_successes / method2_trials if method2_trials > 0 else 0.0

    ci1_lower, ci1_upper = wilson_score_interval(
        method1_successes, method1_trials, confidence
    )
    ci2_lower, ci2_upper = wilson_score_interval(
        method2_successes, method2_trials, confidence
    )

    odds_ratio, p_value = fisher_exact_test(
        method1_successes, method1_trials, method2_successes, method2_trials
    )

    return {
        "method1": {
            "name": method1_name,
            "success_rate": rate1,
            "ci_lower": ci1_lower,
            "ci_upper": ci1_upper,
            "successes": method1_successes,
            "trials": method1_trials,
        },
        "method2": {
            "name": method2_name,
            "success_rate": rate2,
            "ci_lower": ci2_lower,
            "ci_upper": ci2_upper,
            "successes": method2_successes,
            "trials": method2_trials,
        },
        "difference": {
            "absolute": rate1 - rate2,
            "relative": (rate1 / rate2 - 1) if rate2 > 0 else float("inf"),
        },
        "significance_test": {
            "test": "fisher_exact",
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
        },
    }


def compare_mean_turns(
    method1_name: str,
    method1_turns: list[float],
    method2_name: str,
    method2_turns: list[float],
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compare mean turns (or any continuous metric) between two methods.

    Args:
        method1_name: Name of first method
        method1_turns: Turn counts for successful episodes (method 1)
        method2_name: Name of second method
        method2_turns: Turn counts for successful episodes (method 2)
        confidence: Confidence level for intervals

    Returns:
        Dictionary with means, CIs, effect size, and significance test
    """
    mean1, ci1_lower, ci1_upper = bootstrap_mean_ci(method1_turns, confidence)
    mean2, ci2_lower, ci2_upper = bootstrap_mean_ci(method2_turns, confidence)

    effect_size = cohens_d(method1_turns, method2_turns, pooled=True)
    u_stat, p_value = mann_whitney_u_test(method1_turns, method2_turns)

    return {
        "method1": {
            "name": method1_name,
            "mean": mean1,
            "ci_lower": ci1_lower,
            "ci_upper": ci1_upper,
            "n": len(method1_turns),
        },
        "method2": {
            "name": method2_name,
            "mean": mean2,
            "ci_lower": ci2_lower,
            "ci_upper": ci2_upper,
            "n": len(method2_turns),
        },
        "difference": {
            "absolute": mean1 - mean2,
            "cohens_d": effect_size,
            "effect_interpretation": (
                "negligible" if abs(effect_size) < 0.2
                else "small" if abs(effect_size) < 0.5
                else "medium" if abs(effect_size) < 0.8
                else "large"
            ),
        },
        "significance_test": {
            "test": "mann_whitney_u",
            "u_statistic": u_stat,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
        },
    }


def compute_efficiency(
    success_rate: float, gpu_hours: float
) -> dict[str, float]:
    """Calculate resource-efficiency metrics from success and GPU time.

    This is compute efficiency, not sample efficiency: the denominator is
    GPU hours rather than trajectories, environment transitions, or tokens.

    Args:
        success_rate: Visual success rate (0-1)
        gpu_hours: Total GPU hours spent

    Returns:
        Dictionary with efficiency metrics
    """
    if gpu_hours <= 0:
        return {
            "success_per_gpu_hour": 0.0,
            "gpu_hours_per_success_point": float("inf"),
        }

    return {
        "success_per_gpu_hour": success_rate / gpu_hours,
        "gpu_hours_per_success_point": gpu_hours / max(success_rate, 1e-6),
    }


def load_method_results(
    run_dir: Path,
    *,
    require_evaluation_role: str | None = None,
    environment: str | None = None,
) -> dict[str, Any]:
    """Load key results from a training/evaluation run directory.

    Args:
        run_dir: Path to run directory containing manifest.json, etc.

    Returns:
        Dictionary with extracted metrics
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")

    manifest = json.loads(manifest_path.read_text())
    evaluation_role = manifest.get("evaluation_role")
    if (
        require_evaluation_role is not None
        and evaluation_role != require_evaluation_role
    ):
        raise ValueError(
            f"{run_dir} has evaluation_role={evaluation_role!r}; "
            f"expected {require_evaluation_role!r}"
        )
    run_environment = manifest.get("environment")
    if environment is not None and run_environment != environment:
        raise ValueError(
            f"{run_dir} has environment={run_environment!r}; expected {environment!r}"
        )

    gpu_summary_path = run_dir / "gpu_metrics" / "gpu_summary.json"
    gpu_summary = (
        json.loads(gpu_summary_path.read_text())
        if gpu_summary_path.exists()
        else {}
    )

    trajectories: list[dict[str, Any]] = []
    if evaluation_role is not None:
        from vagen.analysis.analyze_rollouts import collect_evaluation_episodes

        trajectories = collect_evaluation_episodes(run_dir)
        successes = [trajectory for trajectory in trajectories if trajectory["success"]]
    else:
        validation_dir = run_dir / "validation"
        if validation_dir.exists():
            latest_jsonl = max(
                validation_dir.glob("*.jsonl"),
                key=lambda p: int(p.stem) if p.stem.isdigit() else -1,
                default=None,
            )
            if latest_jsonl:
                with latest_jsonl.open() as f:
                    for line in f:
                        if line.strip():
                            trajectories.append(json.loads(line))
        successes = [trajectory for trajectory in trajectories if trajectory.get("traj_success")]
    success_rate = len(successes) / len(trajectories) if trajectories else None
    mean_turns = (
        float(np.mean([t.get("num_turns", 0) for t in successes]))
        if successes
        else None
    )

    return {
        "method": manifest.get("source_method", manifest.get("method")),
        "environment": run_environment,
        "seed": manifest.get("source_train_seed", manifest.get("seed")),
        "evaluation_role": evaluation_role,
        "run_dir": str(run_dir),
        "success_rate": success_rate,
        "mean_turns": mean_turns,
        "gpu_hours": gpu_summary.get("gpu_hours"),
        "peak_vram_mib": gpu_summary.get("peak_vram_mib"),
        "trajectories": len(trajectories),
        "successful_trajectories": len(successes),
        "turn_values": [t.get("num_turns") for t in successes if t.get("num_turns")],
    }


def compare_methods_from_runs(
    method1_runs: list[Path],
    method2_runs: list[Path],
    method1_name: str | None = None,
    method2_name: str | None = None,
) -> dict[str, Any]:
    """Compare two methods across multiple runs (seeds).

    Args:
        method1_runs: List of run directories for method 1
        method2_runs: List of run directories for method 2
        method1_name: Optional name override for method 1
        method2_name: Optional name override for method 2

    Returns:
        Comprehensive comparison with success rates, mean turns, efficiency
    """
    results1 = [load_method_results(run) for run in method1_runs]
    results2 = [load_method_results(run) for run in method2_runs]

    name1 = method1_name or results1[0]["method"]
    name2 = method2_name or results2[0]["method"]

    # Success rate comparison
    total_success1 = sum(r["successful_trajectories"] for r in results1)
    total_trials1 = sum(r["trajectories"] for r in results1)
    total_success2 = sum(r["successful_trajectories"] for r in results2)
    total_trials2 = sum(r["trajectories"] for r in results2)

    success_comparison = compare_success_rates(
        name1, total_success1, total_trials1,
        name2, total_success2, total_trials2,
    )

    # Mean turns comparison (successful episodes only)
    all_turns1 = [turn for r in results1 for turn in r["turn_values"]]
    all_turns2 = [turn for r in results2 for turn in r["turn_values"]]

    turns_comparison = (
        compare_mean_turns(name1, all_turns1, name2, all_turns2)
        if all_turns1 and all_turns2
        else None
    )

    # Compute/resource efficiency. This is deliberately not called sample
    # efficiency because no interaction or generated-token denominator is used.
    avg_success1 = total_success1 / total_trials1 if total_trials1 > 0 else 0.0
    avg_success2 = total_success2 / total_trials2 if total_trials2 > 0 else 0.0
    avg_gpu_hours1 = np.mean([r["gpu_hours"] for r in results1 if r["gpu_hours"]])
    avg_gpu_hours2 = np.mean([r["gpu_hours"] for r in results2 if r["gpu_hours"]])

    efficiency1 = compute_efficiency(avg_success1, avg_gpu_hours1)
    efficiency2 = compute_efficiency(avg_success2, avg_gpu_hours2)

    return {
        "methods": {"method1": name1, "method2": name2},
        "runs": {"method1": len(results1), "method2": len(results2)},
        "success_rate_comparison": success_comparison,
        "mean_turns_comparison": turns_comparison,
        "compute_efficiency": {
            "method1": efficiency1,
            "method2": efficiency2,
        },
        "interpretation": _interpret_comparison(
            success_comparison, turns_comparison, efficiency1, efficiency2
        ),
    }


def _interpret_comparison(
    success_comp: dict[str, Any],
    turns_comp: dict[str, Any] | None,
    efficiency1: dict[str, float],
    efficiency2: dict[str, float],
) -> str:
    """Generate human-readable interpretation of method comparison."""
    m1 = success_comp["method1"]["name"]
    m2 = success_comp["method2"]["name"]

    success_sig = success_comp["significance_test"]["significant_at_0.05"]
    success_diff = success_comp["difference"]["absolute"]

    parts = []

    if success_sig:
        winner = m1 if success_diff > 0 else m2
        parts.append(
            f"{winner} has significantly higher success rate "
            f"(p={success_comp['significance_test']['p_value']:.3f})"
        )
    else:
        parts.append("No significant difference in success rates")

    if turns_comp and turns_comp["significance_test"]["significant_at_0.05"]:
        turns_diff = turns_comp["difference"]["absolute"]
        more_efficient = m1 if turns_diff < 0 else m2
        effect = turns_comp["difference"]["effect_interpretation"]
        parts.append(
            f"{more_efficient} achieves success with fewer turns "
            f"(effect: {effect}, p={turns_comp['significance_test']['p_value']:.3f})"
        )

    eff_ratio = (
        efficiency1["success_per_gpu_hour"] / efficiency2["success_per_gpu_hour"]
        if efficiency2["success_per_gpu_hour"] > 0
        else float("inf")
    )
    if eff_ratio > 1.2:
        parts.append(f"{m1} is {eff_ratio:.1f}× more compute-efficient")
    elif eff_ratio < 0.833:
        parts.append(f"{m2} is {1/eff_ratio:.1f}× more compute-efficient")

    return "; ".join(parts) if parts else "Methods are comparable"


if __name__ == "__main__":
    # Example usage
    print("Statistical Analysis Tools for VLM Agent RL")
    print("=" * 60)
    print("\nExample: Compare success rates")
    comparison = compare_success_rates(
        "concat GRPO", 45, 100,
        "no-concat episode GRPO", 52, 100,
    )
    print(json.dumps(comparison, indent=2))
