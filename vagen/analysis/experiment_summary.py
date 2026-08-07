"""Generate comprehensive experiment summaries for publication and reporting.

Combines results from multiple runs into structured summaries suitable for
papers, presentations, and experiment documentation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from vagen.analysis.analyze_rollouts import build_result_row
from vagen.analysis.result_aggregation import aggregate_experiment_matrix


def generate_experiment_summary(
    experiment_dir: Path,
    include_plots: bool = True,
    *,
    evaluation_role: str | None = None,
) -> dict[str, Any]:
    """Generate a comprehensive summary of an experiment directory.

    Args:
        experiment_dir: Path to experiment directory containing multiple runs
        include_plots: Whether to generate visualization plots

    Returns:
        Structured summary dictionary

    Example:
        >>> summary = generate_experiment_summary(Path("exps/vlm_agent_rl"))
        >>> print(summary["key_findings"])
    """
    # Find all run directories
    run_dirs = [
        d for d in experiment_dir.iterdir()
        if d.is_dir()
        and (d / "manifest.json").exists()
        and (
            evaluation_role is None
            or json.loads((d / "manifest.json").read_text()).get(
                "evaluation_role"
            )
            == evaluation_role
        )
    ]

    if not run_dirs:
        raise ValueError(f"No valid run directories found in {experiment_dir}")

    # Aggregate by method
    matrix = aggregate_experiment_matrix(
        run_dirs,
        group_by="method",
        require_evaluation_role=evaluation_role,
    )

    # Build result table
    result_rows = [build_result_row(run_dir) for run_dir in run_dirs]

    # Extract key statistics
    complete_runs = [r for r in result_rows if r["Status"] == "complete"]
    failed_runs = [r for r in result_rows if r["Status"] == "failed"]
    pending_runs = [r for r in result_rows if r["Status"] not in {"complete", "failed"}]

    # Identify best method
    best_method = None
    best_success_rate = 0.0
    for method, data in matrix["groups"].items():
        if isinstance(data, dict) and "success_rate" in data:
            success_rate = data["success_rate"].get("pooled", 0.0)
            if success_rate and success_rate > best_success_rate:
                best_success_rate = success_rate
                best_method = method

    # Compute resource usage
    total_gpu_hours = sum(
        float(r["GPU·h"]) for r in result_rows
        if r["GPU·h"] is not None
    )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment_dir": str(experiment_dir),
        "evaluation_role": evaluation_role,
        "overview": {
            "total_runs": len(run_dirs),
            "complete_runs": len(complete_runs),
            "failed_runs": len(failed_runs),
            "pending_runs": len(pending_runs),
            "unique_methods": matrix["n_groups"],
            "total_gpu_hours": total_gpu_hours,
        },
        "best_method": {
            "name": best_method,
            "success_rate": best_success_rate,
        } if best_method else None,
        "methods": matrix["groups"],
        "result_table": result_rows,
        "key_findings": _extract_key_findings(matrix, result_rows),
    }

    # Generate plots if requested
    if include_plots:
        from vagen.analysis.visualize_results import generate_all_plots

        plots_dir = experiment_dir / "analysis_plots"
        try:
            # Extract results for plotting
            results_for_plotting = []
            for method, data in matrix["groups"].items():
                if isinstance(data, dict) and "success_rate" in data:
                    results_for_plotting.append({
                        "method": method,
                        "success_rate": data["success_rate"].get("pooled", 0.0),
                        "successes": data["success_rate"].get("total_successes", 0),
                        "trials": data["success_rate"].get("total_trials", 1),
                        "gpu_hours": data["gpu_hours"].get("total", 0.0),
                    })

            generate_all_plots(run_dirs, plots_dir, results_for_plotting)
            summary["plots_dir"] = str(plots_dir)
        except Exception as e:
            summary["plots_error"] = str(e)

    return summary


def _extract_key_findings(
    matrix: dict[str, Any],
    result_rows: list[dict[str, Any]],
) -> list[str]:
    """Extract key findings from experiment results."""
    findings = []

    # Check completion rate
    complete = sum(1 for r in result_rows if r["Status"] == "complete")
    total = len(result_rows)
    if complete < total * 0.5:
        findings.append(f"WARNING: Only {complete}/{total} runs completed successfully")

    # Check for parity failures
    failed_parity = sum(
        1 for r in result_rows
        if r["Status"] == "failed" and "parity" in str(r.get("Evidence", "")).lower()
    )
    if failed_parity > 0:
        findings.append(f"CRITICAL: {failed_parity} runs failed parity gate")

    # Method comparison
    methods_with_data = {
        method: data
        for method, data in matrix["groups"].items()
        if isinstance(data, dict) and "success_rate" in data
    }

    if len(methods_with_data) >= 2:
        # Find best and worst
        success_rates = {
            method: data["success_rate"].get("pooled", 0.0)
            for method, data in methods_with_data.items()
            if data["success_rate"].get("pooled") is not None
        }

        if success_rates:
            best = max(success_rates.items(), key=lambda x: x[1])
            worst = min(success_rates.items(), key=lambda x: x[1])

            findings.append(
                f"Best method: {best[0]} ({best[1]:.1%} success)"
            )

            if best[1] - worst[1] > 0.1:
                findings.append(
                    f"Large performance gap: {best[1] - worst[1]:.1%} between best and worst"
                )
            else:
                findings.append("Methods show similar performance")

    return findings


def generate_publication_table(
    summary: dict[str, Any],
    output_path: Path,
    format: str = "latex",
) -> str:
    """Generate publication-ready table from experiment summary.

    Args:
        summary: Output from generate_experiment_summary
        output_path: Path to save table
        format: Output format ("latex" or "markdown")

    Returns:
        Formatted table string

    Example:
        >>> summary = generate_experiment_summary(Path("exps/vlm_agent_rl"))
        >>> table = generate_publication_table(summary, Path("results/table.tex"))
    """
    if format == "latex":
        return _generate_latex_table(summary, output_path)
    elif format == "markdown":
        return _generate_markdown_table(summary, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")


def _generate_markdown_table(
    summary: dict[str, Any],
    output_path: Path,
) -> str:
    """Generate markdown table."""
    lines = [
        "| Method | Success Rate | Mean Turns | Peak VRAM (MiB) | GPU Hours | Status |",
        "|--------|--------------|------------|-----------------|-----------|--------|",
    ]

    for method, data in sorted(summary["methods"].items()):
        if not isinstance(data, dict):
            continue

        success = data.get("success_rate", {})
        turns = data.get("mean_turns", {})
        vram = data.get("peak_vram_mib", {})
        gpu = data.get("gpu_hours", {})

        success_str = (
            f"{success.get('pooled', 0):.3f} ± {success.get('std', 0):.3f}"
            if success.get("pooled") is not None
            else "—"
        )
        turns_str = (
            f"{turns.get('pooled_all_episodes', 0):.1f}"
            if turns.get("pooled_all_episodes") is not None
            else "—"
        )
        vram_str = f"{vram.get('max', 0):.0f}" if vram.get("max") else "—"
        gpu_str = f"{gpu.get('total', 0):.1f}" if gpu.get("total") else "—"

        # Determine status
        status = "✓" if success.get("pooled", 0) > 0 else "○"

        lines.append(
            f"| {method} | {success_str} | {turns_str} | {vram_str} | {gpu_str} | {status} |"
        )

    markdown = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)
    return markdown


def _generate_latex_table(
    summary: dict[str, Any],
    output_path: Path,
) -> str:
    """Generate LaTeX table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Experimental Results}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Success Rate & Mean Turns & Peak VRAM (MiB) & GPU Hours \\",
        r"\midrule",
    ]

    for method, data in sorted(summary["methods"].items()):
        if not isinstance(data, dict):
            continue

        success = data.get("success_rate", {})
        turns = data.get("mean_turns", {})
        vram = data.get("peak_vram_mib", {})
        gpu = data.get("gpu_hours", {})

        # Format with proper LaTeX escaping
        method_tex = method.replace("_", r"\_")

        success_mean = success.get("pooled", 0)
        success_std = success.get("std", 0)
        success_str = (
            f"${success_mean:.3f} \\pm {success_std:.3f}$"
            if success_mean is not None
            else "—"
        )

        turns_str = (
            f"{turns.get('pooled_all_episodes', 0):.1f}"
            if turns.get("pooled_all_episodes") is not None
            else "—"
        )
        vram_str = f"{vram.get('max', 0):.0f}" if vram.get("max") else "—"
        gpu_str = f"{gpu.get('total', 0):.1f}" if gpu.get("total") else "—"

        lines.append(
            f"{method_tex} & {success_str} & {turns_str} & {vram_str} & {gpu_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex)
    return latex


def generate_executive_summary(
    summary: dict[str, Any],
    output_path: Path,
) -> str:
    """Generate executive summary markdown report.

    Args:
        summary: Output from generate_experiment_summary
        output_path: Path to save summary

    Returns:
        Executive summary markdown string

    Example:
        >>> summary = generate_experiment_summary(Path("exps/vlm_agent_rl"))
        >>> exec_summary = generate_executive_summary(summary, Path("results/summary.md"))
    """
    lines = [
        "# Experiment Executive Summary",
        "",
        f"**Generated:** {summary['generated_at']}",
        "",
        "## Overview",
        "",
        f"- **Total Runs:** {summary['overview']['total_runs']}",
        f"- **Completed:** {summary['overview']['complete_runs']}",
        f"- **Failed:** {summary['overview']['failed_runs']}",
        f"- **Pending:** {summary['overview']['pending_runs']}",
        f"- **Methods Evaluated:** {summary['overview']['unique_methods']}",
        f"- **Total GPU Hours:** {summary['overview']['total_gpu_hours']:.1f}",
        "",
    ]

    # Best method
    if summary.get("best_method"):
        best = summary["best_method"]
        lines.extend([
            "## Key Result",
            "",
            f"**Best Method:** {best['name']}",
            f"- Success Rate: {best['success_rate']:.1%}",
            "",
        ])

    # Key findings
    if summary.get("key_findings"):
        lines.extend([
            "## Key Findings",
            "",
        ])
        for finding in summary["key_findings"]:
            lines.append(f"- {finding}")
        lines.append("")

    # Method comparison
    lines.extend([
        "## Method Comparison",
        "",
    ])

    for method, data in sorted(summary["methods"].items()):
        if not isinstance(data, dict):
            continue

        lines.append(f"### {method}")
        lines.append("")

        success = data.get("success_rate", {})
        if success.get("pooled") is not None:
            lines.append(f"- **Success Rate:** {success['pooled']:.1%} (±{success.get('std', 0):.1%}, n={success.get('total_trials', 0)})")

        turns = data.get("mean_turns", {})
        if turns.get("pooled_all_episodes") is not None:
            lines.append(f"- **Mean Turns:** {turns['pooled_all_episodes']:.1f}")

        gpu = data.get("gpu_hours", {})
        if gpu.get("total") is not None:
            lines.append(f"- **Total GPU Hours:** {gpu['total']:.1f}")

        lines.append("")

    markdown = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)
    return markdown


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX table")

    args = parser.parse_args()

    print(f"Analyzing experiment directory: {args.experiment_dir}")
    summary = generate_experiment_summary(
        args.experiment_dir,
        include_plots=not args.no_plots,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON summary
    json_path = args.output_dir / "experiment_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"JSON summary saved to {json_path}")

    # Generate executive summary
    exec_path = args.output_dir / "executive_summary.md"
    generate_executive_summary(summary, exec_path)
    print(f"Executive summary saved to {exec_path}")

    # Generate publication table
    if args.latex:
        table_path = args.output_dir / "results_table.tex"
        generate_publication_table(summary, table_path, format="latex")
        print(f"LaTeX table saved to {table_path}")
    else:
        table_path = args.output_dir / "results_table.md"
        generate_publication_table(summary, table_path, format="markdown")
        print(f"Markdown table saved to {table_path}")

    print("\nKey Findings:")
    for finding in summary.get("key_findings", []):
        print(f"  - {finding}")
