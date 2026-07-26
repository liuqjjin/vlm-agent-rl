"""Run the deterministic CPU evidence suite and write traceable artifacts."""

from __future__ import annotations

import argparse
import asyncio
import csv
import html
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_value_loss
from vagen.custom_advantage.no_concat_episode_grpo import (
    compute_policy_weights,
    trajectory_reward_from_turns,
)
from vagen.envs.sokoban.patch_sokoban_env import get_shortest_action_path
from vagen.envs.sokoban.sokoban_env import Sokoban


ACTION_NAMES = {1: "up", 2: "down", 3: "left", 4: "right"}
SEED = 20260727


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _metadata() -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_available": torch.cuda.is_available(),
        "seed": SEED,
    }


def _value_optimization(use_value_mask: bool, steps: int = 20) -> list[dict[str, Any]]:
    returns = torch.tensor([[-100.0, 2.0]])
    value_mask = returns.ne(-100.0).float()
    predictions = torch.nn.Parameter(torch.tensor([[0.5, -1.0]]))
    optimizer = torch.optim.SGD([predictions], lr=0.2)
    records: list[dict[str, Any]] = []

    for step in range(steps + 1):
        records.append(
            {
                "variant": "masked" if use_value_mask else "legacy_unmasked",
                "step": step,
                "ignored_prediction": float(predictions.detach()[0, 0]),
                "supervised_prediction": float(predictions.detach()[0, 1]),
            }
        )
        if step == steps:
            break
        optimizer.zero_grad()
        loss, _ = compute_value_loss(
            vpreds=predictions,
            values=predictions.detach(),
            returns=returns,
            response_mask=value_mask if use_value_mask else torch.ones_like(value_mask),
            cliprange_value=1000.0,
            loss_agg_mode="token-mean",
        )
        loss.backward()
        records[-1]["ignored_gradient"] = float(predictions.grad[0, 0])
        records[-1]["supervised_gradient"] = float(predictions.grad[0, 1])
        records[-1]["loss"] = float(loss.detach())
        optimizer.step()
    return records


def _response(actions: list[int]) -> str:
    joined = ",".join(ACTION_NAMES[int(action)] for action in actions)
    return f"<think>follow the verified path</think><answer>{joined}</answer>"


async def _run_sokoban_solution(seed: int, actions_per_turn: int) -> dict[str, Any]:
    env = Sokoban(
        {
            "render_mode": "text",
            "dim_room": (6, 6),
            "num_boxes": 1,
            "max_steps": 100,
            "max_actions_per_step": 3,
            "prompt_format": "free_think",
            "min_solution_steps": (3, 5),
        }
    )
    try:
        await env.reset(seed=seed)
        solution = [
            int(action)
            for action in get_shortest_action_path(
                env.env.room_fixed,
                env.env.room_state,
                MAX_DEPTH=20,
            )
        ]
        turn_rewards: list[float] = []
        success = False
        done = False
        for start in range(0, len(solution), actions_per_turn):
            _, reward, done, info = await env.step(
                _response(solution[start : start + actions_per_turn])
            )
            turn_rewards.append(float(reward))
            success = bool(info["success"])
            if done:
                break
        if not success or not done:
            raise RuntimeError(f"verified shortest path did not solve seed {seed}")

        mode_scores = {
            mode: trajectory_reward_from_turns(
                torch.tensor(turn_rewards),
                success,
                mode=mode,
                success_reward=1.0,
                process_reward_cap=0.2,
                format_reward=0.1,
            )
            for mode in ("outcome", "bounded_process", "format_gate")
        }
        return {
            "seed": seed,
            "packing": "split" if actions_per_turn == 1 else "packed",
            "actions_per_turn": actions_per_turn,
            "solution_steps": len(solution),
            "turns": len(turn_rewards),
            "success": success,
            "turn_rewards": turn_rewards,
            "environment_reward": sum(turn_rewards),
            "trajectory_scores": mode_scores,
        }
    finally:
        await env.close()


async def _reward_bias_experiment(seeds: range) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in seeds:
        records.append(await _run_sokoban_solution(seed, actions_per_turn=3))
        records.append(await _run_sokoban_solution(seed, actions_per_turn=1))
    return records


def _policy_mass_experiment() -> list[dict[str, Any]]:
    # Two trajectories in one prompt group:
    # A has turns with 2 and 8 tokens; B has one 4-token turn.
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    groups = np.array(["g", "g", "g"], dtype=object)
    trajectories = np.array([0, 0, 1])
    turns = np.array([1, 2, 1])
    records: list[dict[str, Any]] = []
    for mode in ("token", "turn", "trajectory"):
        weights = compute_policy_weights(
            response_mask,
            groups,
            trajectories,
            turns,
            mode=mode,
        )
        for row, (trajectory, turn) in enumerate(zip(trajectories, turns)):
            records.append(
                {
                    "mode": mode,
                    "trajectory": int(trajectory),
                    "turn": int(turn),
                    "tokens": int(response_mask[row].sum()),
                    "weight_mass": float(weights[row].sum()),
                }
            )
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _paired_reward_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), {})[str(record["packing"])] = record
    rows: list[dict[str, Any]] = []
    for seed, pair in sorted(by_seed.items()):
        packed, split = pair["packed"], pair["split"]
        row: dict[str, Any] = {
            "seed": seed,
            "solution_steps": packed["solution_steps"],
            "packed_turns": packed["turns"],
            "split_turns": split["turns"],
            "packed_environment_reward": packed["environment_reward"],
            "split_environment_reward": split["environment_reward"],
            "environment_reward_delta": (
                split["environment_reward"] - packed["environment_reward"]
            ),
        }
        for mode in ("outcome", "bounded_process", "format_gate"):
            row[f"{mode}_delta"] = (
                split["trajectory_scores"][mode]
                - packed["trajectory_scores"][mode]
            )
        rows.append(row)
    return rows


def _polyline(
    points: list[tuple[float, float]],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
) -> str:
    xmin, xmax = x_domain
    ymin, ymax = y_domain
    mapped = [
        (
            x + (px - xmin) / (xmax - xmin) * width,
            y + height - (py - ymin) / (ymax - ymin) * height,
        )
        for px, py in points
    ]
    return " ".join(f"{px:.2f},{py:.2f}" for px, py in mapped)


def _write_svg(
    path: Path,
    value_rows: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
) -> None:
    width, height = 1080, 580
    panel_w, panel_h = 450, 390
    top = 105
    left_a, left_b = 85, 615
    masked = [
        (row["step"], row["ignored_prediction"])
        for row in value_rows
        if row["variant"] == "masked"
    ]
    legacy = [
        (row["step"], row["ignored_prediction"])
        for row in value_rows
        if row["variant"] == "legacy_unmasked"
    ]
    reward_points = [
        (index, row["environment_reward_delta"])
        for index, row in enumerate(reward_rows)
    ]
    max_delta = max(point[1] for point in reward_points)
    reward_ymax = max(0.5, math.ceil(max_delta * 10) / 10)

    def axis(left: int, title: str, subtitle: str, ymin: float, ymax: float) -> str:
        return (
            f'<text x="{left}" y="44" class="title">{html.escape(title)}</text>'
            f'<text x="{left}" y="69" class="subtitle">{html.escape(subtitle)}</text>'
            f'<line x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" '
            f'y2="{top + panel_h}" class="axis"/>'
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}" class="axis"/>'
            f'<text x="{left - 12}" y="{top + 5}" text-anchor="end" class="tick">{ymax:g}</text>'
            f'<text x="{left - 12}" y="{top + panel_h + 5}" text-anchor="end" '
            f'class="tick">{ymin:g}</text>'
        )

    circles = []
    for index, value in reward_points:
        cx = left_b + index / max(1, len(reward_points) - 1) * panel_w
        cy = top + panel_h - value / reward_ymax * panel_h
        circles.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5" class="point"/>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .bg {{ fill: #fbfaf7; }}
  .title {{ font: 700 20px -apple-system, BlinkMacSystemFont, sans-serif; fill: #17212b; }}
  .subtitle, .tick, .legend {{ font: 13px -apple-system, BlinkMacSystemFont, sans-serif; fill: #53606d; }}
  .axis {{ stroke: #8b96a1; stroke-width: 1.2; }}
  .masked {{ fill: none; stroke: #087e8b; stroke-width: 3; }}
  .legacy {{ fill: none; stroke: #d1495b; stroke-width: 3; }}
  .point {{ fill: #f18f01; opacity: .85; }}
</style>
<rect width="{width}" height="{height}" class="bg"/>
{axis(left_a, "Sparse critic supervision", "Ignored prediction over 20 SGD updates", -100, 1)}
<polyline points="{_polyline(masked, x=left_a, y=top, width=panel_w, height=panel_h, x_domain=(0, 20), y_domain=(-100, 1))}" class="masked"/>
<polyline points="{_polyline(legacy, x=left_a, y=top, width=panel_w, height=panel_h, x_domain=(0, 20), y_domain=(-100, 1))}" class="legacy"/>
<text x="{left_a}" y="{top + panel_h + 31}" class="tick">update 0</text>
<text x="{left_a + panel_w}" y="{top + panel_h + 31}" text-anchor="end" class="tick">update 20</text>
<line x1="{left_a}" y1="548" x2="{left_a + 28}" y2="548" class="masked"/>
<text x="{left_a + 36}" y="553" class="legend">value_mask preserved</text>
<line x1="{left_a + 220}" y1="548" x2="{left_a + 248}" y2="548" class="legacy"/>
<text x="{left_a + 256}" y="553" class="legend">legacy mask dropped</text>
{axis(left_b, "Sokoban length bias", "Extra environment reward from splitting the same solution", 0, reward_ymax)}
{''.join(circles)}
<text x="{left_b}" y="{top + panel_h + 31}" class="tick">seed {reward_rows[0]["seed"]}</text>
<text x="{left_b + panel_w}" y="{top + panel_h + 31}" text-anchor="end" class="tick">seed {reward_rows[-1]["seed"]}</text>
<text x="{left_b}" y="553" class="legend">Each point compares identical shortest-path actions; only turn packing changes.</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


async def _run(output_dir: Path, seed_start: int, seed_count: int) -> dict[str, Any]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    metadata = _metadata()
    value_rows = _value_optimization(True) + _value_optimization(False)
    reward_records = await _reward_bias_experiment(
        range(seed_start, seed_start + seed_count)
    )
    reward_pairs = _paired_reward_rows(reward_records)
    policy_rows = _policy_mass_experiment()

    raw_dir = output_dir / "raw"
    _write_csv(raw_dir / "value_mask_steps.csv", value_rows)
    _write_jsonl(raw_dir / "sokoban_reward_trajectories.jsonl", reward_records)
    _write_csv(raw_dir / "sokoban_reward_pairs.csv", reward_pairs)
    _write_csv(raw_dir / "policy_weight_mass.csv", policy_rows)

    masked_final = next(
        row for row in reversed(value_rows) if row["variant"] == "masked"
    )
    legacy_final = next(
        row for row in reversed(value_rows) if row["variant"] == "legacy_unmasked"
    )
    reward_deltas = [row["environment_reward_delta"] for row in reward_pairs]
    mode_deltas = {
        mode: [row[f"{mode}_delta"] for row in reward_pairs]
        for mode in ("outcome", "bounded_process", "format_gate")
    }
    summary = {
        "metadata": metadata,
        "value_mask": {
            "steps": 20,
            "masked_ignored_initial": value_rows[0]["ignored_prediction"],
            "masked_ignored_final": masked_final["ignored_prediction"],
            "masked_supervised_final": masked_final["supervised_prediction"],
            "legacy_ignored_final": legacy_final["ignored_prediction"],
        },
        "sokoban_reward_bias": {
            "seed_start": seed_start,
            "seed_count": seed_count,
            "all_solved": all(record["success"] for record in reward_records),
            "mean_environment_reward_delta": float(np.mean(reward_deltas)),
            "min_environment_reward_delta": float(np.min(reward_deltas)),
            "max_environment_reward_delta": float(np.max(reward_deltas)),
            "positive_delta_fraction": float(np.mean(np.asarray(reward_deltas) > 0)),
            "mean_score_delta_by_mode": {
                mode: float(np.mean(values)) for mode, values in mode_deltas.items()
            },
        },
        "policy_weighting": {
            mode: {
                str(trajectory): float(
                    sum(
                        row["weight_mass"]
                        for row in policy_rows
                        if row["mode"] == mode and row["trajectory"] == trajectory
                    )
                )
                for trajectory in (0, 1)
            }
            for mode in ("token", "turn", "trajectory")
        },
        "scope": (
            "CPU correctness evidence only. Visual success, VRAM, GPU-hours, "
            "rollout/train ratio, and base-policy state-relative preflight require CUDA."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    summary_rows = [
        {
            "metric": "masked ignored value after 20 updates",
            "value": masked_final["ignored_prediction"],
            "unit": "value",
            "evidence": "raw/value_mask_steps.csv",
        },
        {
            "metric": "legacy ignored value after 20 updates",
            "value": legacy_final["ignored_prediction"],
            "unit": "value",
            "evidence": "raw/value_mask_steps.csv",
        },
        {
            "metric": "mean extra reward from split turns",
            "value": float(np.mean(reward_deltas)),
            "unit": "reward",
            "evidence": "raw/sokoban_reward_pairs.csv",
        },
        {
            "metric": "seeds with positive split-turn bias",
            "value": float(np.mean(np.asarray(reward_deltas) > 0)),
            "unit": "fraction",
            "evidence": "raw/sokoban_reward_pairs.csv",
        },
    ]
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_svg(output_dir / "cpu_diagnostics.svg", value_rows, reward_pairs)

    worst_pair = max(reward_pairs, key=lambda row: row["environment_reward_delta"])
    failures = [
        {
            "id": "critic-sentinel-supervision",
            "kind": "correctness",
            "evidence": {
                "legacy_ignored_final": legacy_final["ignored_prediction"],
                "masked_ignored_final": masked_final["ignored_prediction"],
                "updates": 20,
            },
            "consequence": (
                "Dropping value_mask trains the deliberately ignored position "
                "toward the -100 sentinel."
            ),
        },
        {
            "id": "sokoban-turn-splitting",
            "kind": "reward-hacking",
            "evidence": worst_pair,
            "consequence": (
                "The same successful shortest path receives more environment "
                "reward when emitted over more turns."
            ),
        },
    ]
    _write_jsonl(output_dir / "failure_cases.jsonl", failures)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cpu/local"),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    args = parser.parse_args()
    summary = asyncio.run(_run(args.output_dir, args.seed_start, args.seed_count))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
