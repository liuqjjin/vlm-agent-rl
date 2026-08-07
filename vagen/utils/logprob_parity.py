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

    Args:
        response_mask: Binary mask of shape (batch_size, seq_len) indicating response tokens.
        group_idx: Array-like of group identifiers, one per batch row (supports UUIDs).
        traj_idx: Array-like of trajectory indices within each group, one per batch row.
        turn_idx: Array-like of turn indices within each trajectory, one per batch row.

    Returns:
        torch.Tensor: Deduplicated response mask of shape (batch_size, seq_len) where
            duplicate (group, trajectory, turn) rows are zeroed out.

    Raises:
        ValueError: If response_mask is not rank 2 or index arrays have wrong length.

    Example:
        >>> # Batch with 3 rows, where rows 0 and 2 are padding duplicates
        >>> response_mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 0]])
        >>> group_idx = [0, 1, 0]
        >>> traj_idx = [0, 0, 0]
        >>> turn_idx = [1, 1, 1]
        >>> deduped = deduplicate_turn_response_mask(response_mask, group_idx, traj_idx, turn_idx)
        >>> # deduped[2] will be all zeros (duplicate of row 0)

    Note:
        This function ensures that rollout/train log-probability parity checks are not
        distorted by padding artifacts introduced during distributed data parallelism.
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
    """Summarize ``pi_train(a|s) / pi_rollout(a|s)`` on valid action tokens.

    Computes the ratio between training-forward and rollout-engine log probabilities
    to detect policy staleness, numerical issues, or implementation bugs before
    performing policy updates.

    Args:
        train_log_probs: Log probabilities from training-forward pass, shape (batch_size, seq_len).
        rollout_log_probs: Log probabilities from rollout engine, shape (batch_size, seq_len).
        response_mask: Binary mask indicating valid response tokens, shape (batch_size, seq_len).
        clip_low: Lower bound for "clipped" ratio classification (default: 0.8).
        clip_high: Upper bound for "clipped" ratio classification (default: 1.2).

    Returns:
        dict[str, float | int]: Parity metrics containing:
            - ratio_mean: Mean of exp(train - rollout)
            - ratio_median: Median ratio
            - ratio_p95: 95th percentile ratio
            - ratio_p99: 99th percentile ratio
            - mean_abs_logprob_delta: Mean absolute log-probability difference
            - pre_update_clip_fraction: Fraction of tokens with ratio outside [clip_low, clip_high]
            - num_tokens: Number of valid response tokens analyzed

    Raises:
        ValueError: If shapes don't match, no valid tokens exist, clip bounds are invalid,
            or non-finite values are detected.

    Example:
        >>> train_lp = torch.tensor([[-0.5, -0.6], [-0.7, -0.8]])
        >>> rollout_lp = torch.tensor([[-0.51, -0.59], [-0.71, -0.79]])
        >>> mask = torch.ones_like(train_lp)
        >>> metrics = calculate_rollout_train_parity(train_lp, rollout_lp, mask)
        >>> metrics["ratio_mean"]  # Should be close to 1.0
        >>> metrics["num_tokens"]  # 4

    References:
        This sanity check prevents training on stale rollout data, which causes policy
        degradation in online RL. Similar to PPO's ratio clipping but applied as a
        pre-update gate rather than gradient manipulation.

    Note:
        Ratios are clamped to [-80, 80] in log-space to prevent overflow in quantile
        computation while still detecting severe divergence. A ratio this large indicates
        hard failure regardless of configured thresholds.
    """
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
    """Stop before an update when rollout/training policies clearly disagree.

    Enforces hard thresholds on parity metrics to prevent training on stale or
    corrupted rollout data. Raises an exception if any threshold is violated.

    Args:
        metrics: Dictionary from calculate_rollout_train_parity containing parity statistics.
        max_p95_ratio_deviation: Maximum allowed absolute deviation of P95 ratio from 1.0
            (default: None, disabled). Example: 0.15 means P95 must be in [0.85, 1.15].
        max_p99_ratio_deviation: Maximum allowed absolute deviation of P99 ratio from 1.0
            (default: 0.2). Example: 0.2 means P99 must be in [0.8, 1.2].
        max_mean_abs_logprob_delta: Maximum allowed mean absolute log-probability difference
            (default: 0.1).
        max_clip_fraction: Maximum allowed fraction of tokens with ratio outside configured
            clip bounds (default: 0.05).

    Raises:
        RuntimeError: If any parity threshold is violated, with details of all failures.

    Example:
        >>> metrics = calculate_rollout_train_parity(train_lp, rollout_lp, mask)
        >>> try:
        ...     enforce_rollout_train_parity(metrics, max_p99_ratio_deviation=0.2)
        ... except RuntimeError as e:
        ...     print(f"Parity check failed: {e}")
        ...     # Skip training update, investigate rollout staleness

    References:
        This fail-fast mechanism prevents the policy collapse that occurs when training
        on rollouts from a significantly different policy checkpoint. Common causes include
        checkpoint loading bugs, rollout caching issues, or incorrect data parallelism setup.

    Note:
        When multiple thresholds are violated, all failures are reported in a single
        exception message for complete diagnostics.
    """
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
    """Persist the pre-update parity evidence even when the gate aborts.

    Writes a JSON report containing parity check results for audit trails and debugging.
    Appends to existing reports to build a history of all parity checks across training.

    Args:
        path: Output path for the JSON report file.
        metrics: Parity metrics dictionary from calculate_rollout_train_parity.
        global_step: Training step number for this parity check.
        gate_enabled: Whether the parity gate was active for this check.
        gate_passed: True if all thresholds passed, False if any failed, None if not evaluated.
        thresholds: Dictionary of threshold configuration used for this check.
        error: Optional error message if the parity check raised an exception (default: None).

    Raises:
        ValueError: If the report file exists but is malformed or unreadable.

    Example:
        >>> metrics = calculate_rollout_train_parity(train_lp, rollout_lp, mask)
        >>> try:
        ...     enforce_rollout_train_parity(metrics)
        ...     passed = True
        ...     error = None
        ... except RuntimeError as e:
        ...     passed = False
        ...     error = str(e)
        >>> write_rollout_train_parity_report(
        ...     "parity.json",
        ...     metrics,
        ...     global_step=1000,
        ...     gate_enabled=True,
        ...     gate_passed=passed,
        ...     thresholds={"max_p99_ratio_deviation": 0.2},
        ...     error=error,
        ... )

    Note:
        The report maintains an "attempts" array containing all historical parity checks.
        The top-level fields reflect the most recent check for quick access. This structure
        enables tracking parity degradation over training and diagnosing intermittent failures.
    """
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
