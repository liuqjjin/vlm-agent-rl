"""Generate visualization plots for experimental results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    run_dirs: list[Path],
    metric: str = "success_rate",
    output_path: Path | None = None,
    title: str | None = None,
) -> None:
    """Plot learning curves from validation checkpoints.

    Args:
        run_dirs: List of run directories with validation/*.jsonl files
        metric: Metric to plot (success_rate, mean_turns, mean_reward)
        output_path: Where to save plot (if None, display instead)
        title: Plot title (auto-generated if None)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for run_dir in run_dirs:
        validation_dir = run_dir / "validation"
        if not validation_dir.exists():
            continue

        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            method_name = manifest.get("method", run_dir.name)
            seed = manifest.get("seed", "?")
            label = f"{method_name} (seed {seed})"
        else:
            label = run_dir.name

        steps = []
        values = []

        for jsonl_path in sorted(validation_dir.glob("*.jsonl")):
            try:
                step = int(jsonl_path.stem)
            except ValueError:
                continue

            trajectories = []
            with jsonl_path.open() as f:
                for line in f:
                    if line.strip():
                        trajectories.append(json.loads(line))

            if not trajectories:
                continue

            if metric == "success_rate":
                successes = sum(1 for t in trajectories if t.get("traj_success"))
                value = successes / len(trajectories)
            elif metric == "mean_turns":
                successful = [
                    t.get("num_turns", 0)
                    for t in trajectories
                    if t.get("traj_success") and t.get("num_turns")
                ]
                value = float(np.mean(successful)) if successful else None
            elif metric == "mean_reward":
                value = float(
                    np.mean([t.get("score", 0.0) for t in trajectories])
                )
            else:
                continue

            if value is not None:
                steps.append(step)
                values.append(value)

        if steps:
            ax.plot(steps, values, marker="o", label=label, linewidth=2, markersize=4)

    ax.set_xlabel("Training Step", fontsize=12)
    ylabel = {
        "success_rate": "Success Rate",
        "mean_turns": "Mean Turns (Successful)",
        "mean_reward": "Mean Reward",
    }.get(metric, metric)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title or f"{ylabel} vs Training Steps", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_success_rate_comparison(
    results: list[dict[str, Any]],
    output_path: Path | None = None,
) -> None:
    """Bar plot comparing success rates with confidence intervals.

    Args:
        results: List of dicts with keys: method, success_rate, ci_lower, ci_upper
        output_path: Where to save plot
    """
    from vagen.analysis.statistical_analysis import wilson_score_interval

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    rates = []
    ci_lowers = []
    ci_uppers = []

    for result in results:
        methods.append(result["method"])
        rate = result.get("success_rate", 0.0)
        rates.append(rate)

        # Compute CI if not provided
        if "ci_lower" in result and "ci_upper" in result:
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])
        elif "successes" in result and "trials" in result:
            lower, upper = wilson_score_interval(
                result["successes"], result["trials"], 0.95
            )
            ci_lowers.append(lower)
            ci_uppers.append(upper)
        else:
            ci_lowers.append(rate)
            ci_uppers.append(rate)

    x_pos = np.arange(len(methods))
    errors = [
        [rates[i] - ci_lowers[i] for i in range(len(rates))],
        [ci_uppers[i] - rates[i] for i in range(len(rates))],
    ]

    ax.bar(x_pos, rates, yerr=errors, capsize=5, alpha=0.7, color="steelblue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Success Rate Comparison (95% CI)", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_efficiency_scatter(
    results: list[dict[str, Any]],
    output_path: Path | None = None,
) -> None:
    """Scatter plot of success rate versus GPU-hour resource use.

    Args:
        results: List of dicts with keys: method, success_rate, gpu_hours
        output_path: Where to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        method = result.get("method", "unknown")
        success = result.get("success_rate", 0.0)
        gpu_hours = result.get("gpu_hours", 0.0)

        if gpu_hours > 0:
            ax.scatter(gpu_hours, success, s=100, alpha=0.7, label=method)
            ax.annotate(
                method,
                (gpu_hours, success),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xlabel("GPU Hours", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Compute Efficiency: Success vs GPU Hours", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add diagonal lines for efficiency reference
    max_gpu = max(r.get("gpu_hours", 0) for r in results)
    for efficiency in [0.01, 0.02, 0.05]:
        x = np.linspace(0, max_gpu, 100)
        y = efficiency * x
        ax.plot(
            x, y, "--", alpha=0.3, color="gray",
            label=f"{efficiency:.3f} success/GPU·h"
        )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_reward_distribution(
    run_dirs: list[Path],
    output_path: Path | None = None,
) -> None:
    """Plot reward distributions from latest validation JSONL.

    Args:
        run_dirs: List of run directories
        output_path: Where to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for run_dir in run_dirs:
        validation_dir = run_dir / "validation"
        if not validation_dir.exists():
            continue

        latest_jsonl = max(
            validation_dir.glob("*.jsonl"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else -1,
            default=None,
        )

        if not latest_jsonl:
            continue

        rewards = []
        with latest_jsonl.open() as f:
            for line in f:
                if line.strip():
                    traj = json.loads(line)
                    rewards.append(traj.get("score", 0.0))

        if rewards:
            label = run_dir.name
            ax.hist(rewards, bins=20, alpha=0.5, label=label)

    ax.set_xlabel("Trajectory Reward", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Reward Distribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def generate_all_plots(
    run_dirs: list[Path],
    output_dir: Path,
    results: list[dict[str, Any]] | None = None,
) -> None:
    """Generate all standard plots for a set of runs.

    Args:
        run_dirs: List of run directories
        output_dir: Directory to save all plots
        results: Pre-loaded results (if None, will load from run_dirs)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating learning curves...")
    plot_learning_curves(
        run_dirs,
        metric="success_rate",
        output_path=output_dir / "learning_curve_success.png",
    )
    plot_learning_curves(
        run_dirs,
        metric="mean_turns",
        output_path=output_dir / "learning_curve_turns.png",
    )

    if results:
        print("Generating comparison plots...")
        plot_success_rate_comparison(
            results,
            output_path=output_dir / "success_comparison.png",
        )
        plot_efficiency_scatter(
            results,
            output_path=output_dir / "efficiency_scatter.png",
        )

    print("Generating reward distributions...")
    plot_reward_distribution(
        run_dirs,
        output_path=output_dir / "reward_distribution.png",
    )

    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.run:
        parser.error("provide at least one --run directory")

    generate_all_plots(args.run, args.output_dir)
