"""Preflight gate for state-relative (GiGPO-style) advantages.

The gate intentionally runs before any state-relative training.  If exact
pre-action state groups are mostly singletons, actions do not vary, or their
return-to-go has no variance, a relative advantage is not identifiable.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_THRESHOLDS = {
    "min_rows": 64,
    "max_missing_anchor_fraction": 0.01,
    "min_comparable_row_fraction": 0.20,
    "min_action_diversity_fraction": 0.10,
    "min_mean_return_to_go_variance": 1e-4,
}

ACTION_PATTERN = re.compile(
    r"<(?P<tag>answer|action)\b[^>]*>(?P<body>.*?)</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _id(value: Any) -> tuple[str, str]:
    if isinstance(value, np.generic):
        value = value.item()
    return type(value).__name__, str(value)


def _normalized_action(value: Any) -> str:
    text = str(value or "")
    match = ACTION_PATTERN.search(text)
    if match:
        text = match.group("body")
    return " ".join(text.strip().lower().split())


def _deduplicate(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: dict[tuple[tuple[str, str], int, int], dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = (_id(row["group_idx"]), int(row["traj_idx"]), int(row["turn_idx"]))
        if key not in unique:
            unique[key] = row
            continue
        comparable = ("score", "output", "state_anchor", "last_turn", "traj_success")
        if any(unique[key].get(field) != row.get(field) for field in comparable):
            raise ValueError(f"conflicting duplicate rollout row: {key}")
        duplicates += 1
    return list(unique.values()), duplicates


def analyze_state_relative_rows(
    rows: Iterable[dict[str, Any]],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Measure whether exact state-relative groups carry a learnable signal."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    unique_rows, duplicate_count = _deduplicate(rows)
    row_count = len(unique_rows)

    rows_by_trajectory: dict[tuple[tuple[str, str], int], list[dict[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        rows_by_trajectory[(_id(row["group_idx"]), int(row["traj_idx"]))].append(row)

    return_to_go: dict[tuple[tuple[str, str], int, int], float] = {}
    for trajectory, trajectory_rows in rows_by_trajectory.items():
        trajectory_rows.sort(key=lambda row: int(row["turn_idx"]))
        running = 0.0
        for row in reversed(trajectory_rows):
            running += float(row.get("score", 0.0))
            return_to_go[(trajectory[0], trajectory[1], int(row["turn_idx"]))] = running

    missing_anchor_count = sum(not row.get("state_anchor") for row in unique_rows)
    state_groups: dict[tuple[tuple[str, str], str], list[dict[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        anchor = row.get("state_anchor")
        if anchor:
            state_groups[(_id(row["group_idx"]), str(anchor))].append(row)

    group_sizes = [len(group_rows) for group_rows in state_groups.values()]
    comparable_groups = [group_rows for group_rows in state_groups.values() if len(group_rows) >= 2]
    comparable_rows = sum(len(group_rows) for group_rows in comparable_groups)
    anchored_rows = row_count - missing_anchor_count
    singleton_groups = sum(size == 1 for size in group_sizes)

    diverse_groups = 0
    rtg_variances: list[float] = []
    for group_rows in comparable_groups:
        actions = {_normalized_action(row.get("output")) for row in group_rows}
        if len(actions) >= 2:
            diverse_groups += 1
        returns = [
            return_to_go[(_id(row["group_idx"]), int(row["traj_idx"]), int(row["turn_idx"]))]
            for row in group_rows
        ]
        rtg_variances.append(float(np.var(returns, ddof=0)))

    state_group_count = len(group_sizes)
    comparable_group_count = len(comparable_groups)
    metrics = {
        "rows": row_count,
        "trajectories": len(rows_by_trajectory),
        "padding_duplicates": duplicate_count,
        "missing_anchor_fraction": missing_anchor_count / row_count if row_count else 1.0,
        "state_groups": state_group_count,
        "singleton_group_fraction": singleton_groups / state_group_count if state_group_count else 1.0,
        "comparable_groups": comparable_group_count,
        "comparable_row_fraction": comparable_rows / anchored_rows if anchored_rows else 0.0,
        "action_diversity_fraction": (
            diverse_groups / comparable_group_count if comparable_group_count else 0.0
        ),
        "mean_return_to_go_variance": float(np.mean(rtg_variances)) if rtg_variances else 0.0,
        "group_size_p50": float(np.quantile(group_sizes, 0.50)) if group_sizes else 0.0,
        "group_size_p95": float(np.quantile(group_sizes, 0.95)) if group_sizes else 0.0,
        "group_size_max": max(group_sizes, default=0),
    }

    checks = {
        "enough_rows": metrics["rows"] >= limits["min_rows"],
        "anchors_present": (
            metrics["missing_anchor_fraction"] <= limits["max_missing_anchor_fraction"]
        ),
        "enough_comparable_rows": (
            metrics["comparable_row_fraction"] >= limits["min_comparable_row_fraction"]
        ),
        "actions_vary": (
            metrics["action_diversity_fraction"] >= limits["min_action_diversity_fraction"]
        ),
        "returns_vary": (
            metrics["mean_return_to_go_variance"]
            >= limits["min_mean_return_to_go_variance"]
        ),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "decision": "proceed" if not reasons else "stop",
        "reasons": reasons,
        "thresholds": limits,
        "metrics": metrics,
    }


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="no-concat rollout JSONL files")
    parser.add_argument("--output", type=Path, help="write the preflight report as JSON")
    parser.add_argument("--fail-on-stop", action="store_true", help="exit 2 when the gate says stop")
    args = parser.parse_args()

    report = analyze_state_relative_rows(load_jsonl(args.inputs))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 2 if args.fail_on_stop and report["decision"] == "stop" else 0


if __name__ == "__main__":
    raise SystemExit(main())
