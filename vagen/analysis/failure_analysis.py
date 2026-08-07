"""Enhanced failure analysis for debugging and result interpretation.

Provides deep analysis of failure cases including error patterns, trajectory
characteristics, and actionable debugging insights.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

def classify_failure_reason(
    episode: dict[str, Any],
) -> str:
    """Classify the root cause of episode failure.

    Args:
        episode: Episode dictionary with finish_reason, metrics, etc.

    Returns:
        Failure classification category

    Example:
        >>> episode = {"finish_reason": "max_turns", "num_turns": 20, "cumulative_reward": 0.5}
        >>> classify_failure_reason(episode)
        'timeout_with_partial_progress'
    """
    finish_reason = episode.get("finish_reason", "unknown")
    success = episode.get("success", False)
    num_turns = episode.get("num_turns", 0)
    reward = episode.get("cumulative_reward", 0.0)
    action_checks = episode.get("action_check_count")
    valid_actions = episode.get("valid_action_count")
    invalid_actions = (
        isinstance(action_checks, int)
        and action_checks > 0
        and isinstance(valid_actions, int)
        and valid_actions < action_checks
    )

    if success:
        return "success"

    # Infrastructure errors
    if finish_reason in {"model_error", "env_error", "error"}:
        return "infrastructure_error"

    # Action validity issues
    if invalid_actions:
        if valid_actions == 0:
            return "all_actions_invalid"
        return "frequent_invalid_actions"

    # Timeout cases
    if finish_reason == "max_turns":
        if reward > 0.5:
            return "timeout_with_high_progress"
        elif reward >= 0.1:
            return "timeout_with_partial_progress"
        else:
            return "timeout_with_no_progress"

    # Early termination
    if num_turns < 3:
        return "early_termination"

    # Stalled (many turns, no progress)
    if num_turns > 10 and reward < 0.1:
        return "stalled_no_progress"

    return "other_failure"


def analyze_failure_patterns(
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze patterns in failed episodes to identify systematic issues.

    Args:
        episodes: List of episode dictionaries from evaluation or training

    Returns:
        Dictionary with failure pattern analysis

    Example:
        >>> from vagen.analysis.analyze_rollouts import collect_evaluation_episodes
        >>> episodes = collect_evaluation_episodes(Path("exps/eval_dump"))
        >>> patterns = analyze_failure_patterns(episodes)
        >>> print(f"Most common failure: {patterns['most_common_failure']}")
    """
    failures = [ep for ep in episodes if not ep.get("success", False)]

    if not failures:
        return {
            "total_failures": 0,
            "failure_rate": 0.0,
            "failure_categories": {},
            "most_common_failure": None,
            "infrastructure_error_rate": 0.0,
        }

    # Classify each failure
    categories = Counter()
    for episode in failures:
        category = classify_failure_reason(episode)
        categories[category] += 1

    # Compute rates
    total = len(episodes)
    failure_rate = len(failures) / total if total > 0 else 0.0
    infrastructure_errors = categories.get("infrastructure_error", 0)
    infra_error_rate = infrastructure_errors / total if total > 0 else 0.0

    # Analyze trajectories by failure type
    failure_details = defaultdict(list)
    for episode in failures:
        category = classify_failure_reason(episode)
        failure_details[category].append({
            "rollout_id": episode.get("rollout_id"),
            "seed": episode.get("seed"),
            "num_turns": episode.get("num_turns"),
            "reward": episode.get("cumulative_reward"),
            "finish_reason": episode.get("finish_reason"),
            "transcript_path": episode.get("transcript_path"),
        })

    return {
        "total_episodes": total,
        "total_failures": len(failures),
        "failure_rate": failure_rate,
        "failure_categories": dict(categories),
        "most_common_failure": categories.most_common(1)[0][0] if categories else None,
        "infrastructure_error_rate": infra_error_rate,
        "actionable_failure_rate": (len(failures) - infrastructure_errors) / total if total > 0 else 0.0,
        "failure_details_by_category": dict(failure_details),
    }


def extract_error_messages(
    transcript_paths: list[Path],
    limit: int = 50,
) -> dict[str, Any]:
    """Extract and categorize error messages from episode transcripts.

    Args:
        transcript_paths: List of transcript file paths
        limit: Maximum number of transcripts to process

    Returns:
        Dictionary with error message patterns and frequencies

    Example:
        >>> transcripts = [Path("exps/eval/ep0/transcript.txt")]
        >>> errors = extract_error_messages(transcripts)
        >>> for pattern, count in errors['top_error_patterns']:
        ...     print(f"{pattern}: {count}")
    """
    error_patterns = Counter()
    error_contexts = defaultdict(list)

    error_indicators = [
        "error",
        "exception",
        "traceback",
        "failed",
        "invalid",
        "cannot",
        "timeout",
    ]

    for transcript_path in transcript_paths[:limit]:
        if not transcript_path.exists():
            continue

        try:
            content = transcript_path.read_text()
            lines = content.lower().split("\n")

            for line in lines:
                if any(indicator in line for indicator in error_indicators):
                    # Extract key parts of error
                    # Remove timestamps, paths, and IDs to get pattern
                    pattern = re.sub(r"\d{4}-\d{2}-\d{2}", "", line)
                    pattern = re.sub(r"\b\d+\b", "<N>", pattern)
                    pattern = re.sub(r"/[^\s]+", "<PATH>", pattern)
                    pattern = pattern.strip()[:100]  # Truncate

                    if pattern:
                        error_patterns[pattern] += 1
                        error_contexts[pattern].append(str(transcript_path))

        except Exception:
            continue

    return {
        "n_transcripts_analyzed": min(len(transcript_paths), limit),
        "unique_error_patterns": len(error_patterns),
        "top_error_patterns": error_patterns.most_common(10),
        "error_contexts": {
            pattern: contexts[:3]  # Keep 3 example paths per pattern
            for pattern, contexts in list(error_contexts.items())[:10]
        },
    }


def identify_action_repetition_failures(
    episodes: list[dict[str, Any]],
    repetition_threshold: int = 5,
) -> dict[str, Any]:
    """Identify failures caused by action repetition loops.

    Args:
        episodes: List of episode dictionaries with assistant_texts
        repetition_threshold: Minimum consecutive repetitions to flag

    Returns:
        Dictionary with repetition loop analysis

    Example:
        >>> episodes = [
        ...     {"rollout_id": "ep0", "assistant_texts": ["left", "left", "left", "left", "left"]},
        ... ]
        >>> loops = identify_action_repetition_failures(episodes, repetition_threshold=5)
        >>> print(loops['episodes_with_loops'])
    """
    episodes_with_loops = []

    for episode in episodes:
        texts = episode.get("assistant_texts", [])
        if len(texts) < repetition_threshold:
            continue

        # Extract actions (simple pattern matching)
        actions = []
        for text in texts:
            # Look for common action patterns
            text_lower = text.lower()
            for action in ["up", "down", "left", "right", "forward", "backward", "turn"]:
                if action in text_lower:
                    actions.append(action)
                    break
            else:
                actions.append("unknown")

        # Find consecutive repetitions
        max_repetition = 1
        current_repetition = 1
        repeated_action = None

        for i in range(1, len(actions)):
            if actions[i] == actions[i - 1]:
                current_repetition += 1
                if current_repetition > max_repetition:
                    max_repetition = current_repetition
                    repeated_action = actions[i]
            else:
                current_repetition = 1

        if max_repetition >= repetition_threshold:
            episodes_with_loops.append({
                "rollout_id": episode.get("rollout_id"),
                "seed": episode.get("seed"),
                "max_repetition": max_repetition,
                "repeated_action": repeated_action,
                "num_turns": episode.get("num_turns"),
                "success": episode.get("success", False),
            })

    return {
        "total_episodes": len(episodes),
        "episodes_with_loops": len(episodes_with_loops),
        "loop_rate": len(episodes_with_loops) / len(episodes) if episodes else 0.0,
        "repetition_threshold": repetition_threshold,
        "examples": episodes_with_loops[:10],
    }


def generate_failure_analysis_report(
    failure_patterns: dict[str, Any],
    error_messages: dict[str, Any] | None = None,
    repetition_loops: dict[str, Any] | None = None,
    output_path: Path | None = None,
) -> str:
    """Generate comprehensive failure analysis markdown report.

    Args:
        failure_patterns: Output from analyze_failure_patterns
        error_messages: Optional output from extract_error_messages
        repetition_loops: Optional output from identify_action_repetition_failures
        output_path: Optional path to save report

    Returns:
        Markdown report as string

    Example:
        >>> patterns = analyze_failure_patterns(episodes)
        >>> report = generate_failure_analysis_report(patterns)
        >>> print(report)
    """
    lines = [
        "# Failure Analysis Report",
        "",
        "## Summary",
        "",
        f"- **Total Episodes**: {failure_patterns['total_episodes']}",
        f"- **Total Failures**: {failure_patterns['total_failures']}",
        f"- **Failure Rate**: {failure_patterns['failure_rate']:.1%}",
        f"- **Infrastructure Error Rate**: {failure_patterns['infrastructure_error_rate']:.1%}",
        f"- **Actionable Failure Rate**: {failure_patterns['actionable_failure_rate']:.1%}",
        "",
        "## Failure Categories",
        "",
    ]

    categories = failure_patterns["failure_categories"]
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        rate = count / failure_patterns["total_failures"] if failure_patterns["total_failures"] > 0 else 0.0
        lines.append(f"- **{category}**: {count} ({rate:.1%})")

    lines.extend([
        "",
        f"**Most Common Failure**: {failure_patterns['most_common_failure']}",
        "",
    ])

    # Add error message analysis if provided
    if error_messages:
        lines.extend([
            "## Error Message Patterns",
            "",
            f"Analyzed {error_messages['n_transcripts_analyzed']} transcripts",
            "",
            "### Top Error Patterns",
            "",
        ])

        for i, (pattern, count) in enumerate(error_messages["top_error_patterns"], 1):
            lines.append(f"{i}. `{pattern}` ({count} occurrences)")

        lines.append("")

    # Add repetition loop analysis if provided
    if repetition_loops:
        lines.extend([
            "## Action Repetition Loops",
            "",
            f"- **Episodes with Loops**: {repetition_loops['episodes_with_loops']} / {repetition_loops['total_episodes']}",
            f"- **Loop Rate**: {repetition_loops['loop_rate']:.1%}",
            f"- **Threshold**: {repetition_loops['repetition_threshold']} consecutive actions",
            "",
        ])

        if repetition_loops["examples"]:
            lines.append("### Examples")
            lines.append("")
            for example in repetition_loops["examples"][:5]:
                lines.append(f"- Episode `{example['rollout_id']}`: repeated `{example['repeated_action']}` {example['max_repetition']} times")

        lines.append("")

    # Add actionable recommendations
    lines.extend([
        "## Recommendations",
        "",
    ])

    most_common = failure_patterns["most_common_failure"]
    if most_common == "timeout_with_partial_progress":
        lines.append("- Consider increasing max_turns or adjusting reward shaping")
    elif most_common == "frequent_invalid_actions":
        lines.append("- Review action space and format validation")
        lines.append("- Check prompt clarity for action specification")
    elif most_common == "stalled_no_progress":
        lines.append("- Investigate reward signal - may be too sparse")
        lines.append("- Check for exploration issues")
    elif most_common == "infrastructure_error":
        lines.append("- **CRITICAL**: Fix infrastructure issues before interpreting behavioral results")

    if repetition_loops and repetition_loops["loop_rate"] > 0.1:
        lines.append("- High action repetition detected - consider adding anti-loop penalty")

    markdown = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)

    return markdown


if __name__ == "__main__":
    import argparse
    from vagen.analysis.analyze_rollouts import collect_evaluation_episodes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyze-errors", action="store_true", help="Extract error messages from transcripts")
    parser.add_argument("--check-loops", action="store_true", help="Check for action repetition loops")

    args = parser.parse_args()

    # Load episodes
    print(f"Loading episodes from {args.eval_dump}...")
    episodes = collect_evaluation_episodes(args.eval_dump)
    print(f"Loaded {len(episodes)} episodes")

    # Analyze failure patterns
    print("Analyzing failure patterns...")
    patterns = analyze_failure_patterns(episodes)

    error_messages = None
    if args.analyze_errors:
        print("Extracting error messages...")
        transcript_paths = [Path(ep["transcript_path"]) for ep in episodes if "transcript_path" in ep]
        error_messages = extract_error_messages(transcript_paths)

    repetition_loops = None
    if args.check_loops:
        print("Checking for action repetition loops...")
        repetition_loops = identify_action_repetition_failures(episodes)

    # Generate report
    print("Generating report...")
    report = generate_failure_analysis_report(patterns, error_messages, repetition_loops, args.output)

    # Save JSON
    json_path = args.output.with_suffix(".json")
    json_data = {
        "failure_patterns": patterns,
        "error_messages": error_messages,
        "repetition_loops": repetition_loops,
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))

    print(f"Report saved to {args.output}")
    print(f"JSON data saved to {json_path}")
    print(f"\nFailure rate: {patterns['failure_rate']:.1%}")
    print(f"Most common failure: {patterns['most_common_failure']}")
