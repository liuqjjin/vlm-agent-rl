"""Rollout-engine versus training-forward log-probability sanity checks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def deduplicate_turn_response_mask(
    response_mask: torch.Tensor,
    group_idx: Any,
    traj_idx: Any,
    turn_idx: Any,
) -> torch.Tensor:
    """Mask repeated no-concat rows introduced by distributed padding.

    The trainer pads by copying complete rows before balancing them across data
    parallel ranks. Parity is a property of the sampled policy data, so copied
    ``(group, trajectory, turn)`` rows must not change its token distribution.
    """
    if response_mask.ndim != 2:
        raise ValueError(
            f"response_mask must be rank 2, got shape {tuple(response_mask.shape)}"
        )
    row_count = response_mask.shape[0]

    def _rows(values: Any, name: str) -> list[Any]:
        if isinstance(values, torch.Tensor):
            flattened = values.detach().cpu().reshape(-1).tolist()
        else:
            try:
                flattened = list(values)
            except TypeError as error:
                raise ValueError(f"{name} must contain one value per row") from error
        if len(flattened) != row_count:
            raise ValueError(
                f"{name} has {len(flattened)} rows; expected {row_count}"
            )
        return flattened

    groups = _rows(group_idx, "group_idx")
    trajectories = _rows(traj_idx, "traj_idx")
    turns = _rows(turn_idx, "turn_idx")
    keep = torch.zeros(row_count, dtype=torch.bool, device=response_mask.device)
    seen: set[tuple[tuple[str, str], int, int]] = set()
    for row, (group, trajectory, turn) in enumerate(
        zip(groups, trajectories, turns, strict=True)
    ):
        key = ((type(group).__name__, str(group)), int(trajectory), int(turn))
        if key not in seen:
            seen.add(key)
            keep[row] = True
    return response_mask * keep.unsqueeze(-1).to(dtype=response_mask.dtype)


def calculate_rollout_train_parity(
    train_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_low: float = 0.8,
    clip_high: float = 1.2,
) -> dict[str, float | int]:
    """Summarize ``pi_train(a|s) / pi_rollout(a|s)`` on valid action tokens."""
    if train_log_probs.shape != rollout_log_probs.shape or train_log_probs.shape != response_mask.shape:
        raise ValueError(
            "train_log_probs, rollout_log_probs, and response_mask must have identical shapes; "
            f"got {train_log_probs.shape}, {rollout_log_probs.shape}, {response_mask.shape}"
        )
    if not 0 < clip_low < 1 < clip_high:
        raise ValueError(f"expected 0 < clip_low < 1 < clip_high, got {clip_low}, {clip_high}")

    valid = response_mask.to(dtype=torch.bool)
    if not bool(valid.any()):
        raise ValueError("no valid response tokens for rollout/train parity")
    train = train_log_probs[valid].detach().to(dtype=torch.float64)
    rollout = rollout_log_probs[valid].detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(train).all() and torch.isfinite(rollout).all()):
        raise ValueError("non-finite log probabilities in rollout/train parity inputs")

    delta = train - rollout
    # A delta this large is already a hard failure; clamping only keeps summary
    # quantiles finite and does not make the configured gate pass.
    ratio = torch.exp(delta.clamp(min=-80.0, max=80.0))
    clipped = (ratio < float(clip_low)) | (ratio > float(clip_high))
    return {
        "ratio_mean": float(ratio.mean().item()),
        "ratio_median": float(torch.quantile(ratio, 0.50).item()),
        "ratio_p95": float(torch.quantile(ratio, 0.95).item()),
        "ratio_p99": float(torch.quantile(ratio, 0.99).item()),
        "mean_abs_logprob_delta": float(delta.abs().mean().item()),
        "pre_update_clip_fraction": float(clipped.to(dtype=torch.float64).mean().item()),
        "num_tokens": int(ratio.numel()),
    }


def enforce_rollout_train_parity(
    metrics: Mapping[str, float | int],
    *,
    max_p95_ratio_deviation: float | None = None,
    max_p99_ratio_deviation: float = 0.2,
    max_mean_abs_logprob_delta: float = 0.1,
    max_clip_fraction: float = 0.05,
) -> None:
    """Stop before an update when rollout/training policies clearly disagree."""
    failures: list[str] = []
    if max_p95_ratio_deviation is not None:
        deviation = abs(float(metrics["ratio_p95"]) - 1.0)
        if deviation > float(max_p95_ratio_deviation):
            failures.append(
                f"ratio P95 deviation {deviation:.4f} > {float(max_p95_ratio_deviation):.4f}"
            )
    p99_deviation = abs(float(metrics["ratio_p99"]) - 1.0)
    if p99_deviation > float(max_p99_ratio_deviation):
        failures.append(f"ratio P99 deviation {p99_deviation:.4f} > {float(max_p99_ratio_deviation):.4f}")
    mean_abs_delta = float(metrics["mean_abs_logprob_delta"])
    if mean_abs_delta > float(max_mean_abs_logprob_delta):
        failures.append(
            f"mean absolute logprob delta {mean_abs_delta:.4f} > "
            f"{float(max_mean_abs_logprob_delta):.4f}"
        )
    clip_fraction = float(metrics["pre_update_clip_fraction"])
    if clip_fraction > float(max_clip_fraction):
        failures.append(f"pre-update clip fraction {clip_fraction:.4f} > {float(max_clip_fraction):.4f}")
    if failures:
        raise RuntimeError("rollout/train logprob parity check failed: " + "; ".join(failures))


def write_rollout_train_parity_report(
    path: str | Path,
    metrics: Mapping[str, float | int],
    *,
    global_step: int,
    gate_enabled: bool,
    gate_passed: bool | None,
    thresholds: Mapping[str, Any],
    error: str | None = None,
) -> None:
    """Persist the pre-update parity evidence even when the gate aborts."""
    attempt = {
        "global_step": int(global_step),
        "gate_enabled": bool(gate_enabled),
        "gate_passed": gate_passed,
        "metrics": dict(metrics),
        "thresholds": dict(thresholds),
        "error": error,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    if destination.exists():
        try:
            previous = json.loads(destination.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot append parity evidence to unreadable report: {destination}"
            ) from exc
        if not isinstance(previous, dict):
            raise ValueError(
                f"cannot append parity evidence to non-object report: {destination}"
            )
        prior_attempts = previous.get("attempts")
        if prior_attempts is not None:
            if not isinstance(prior_attempts, list) or not all(
                isinstance(item, dict) for item in prior_attempts
            ):
                raise ValueError(
                    f"cannot append parity evidence to malformed history: {destination}"
                )
            attempts = list(prior_attempts)
        elif "global_step" in previous:
            attempts = [
                {
                    key: previous.get(key)
                    for key in (
                        "global_step",
                        "gate_enabled",
                        "gate_passed",
                        "metrics",
                        "thresholds",
                        "error",
                    )
                }
            ]
    attempts.append(attempt)
    report = {**attempt, "attempts": attempts}
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
