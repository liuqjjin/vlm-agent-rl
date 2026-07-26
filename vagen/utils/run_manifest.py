"""Identity checks for resumable formal experiment directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping


ResumeState = Literal[
    "complete",
    "resumable",
    "failed-parity",
    "tainted-gpu-metrics",
]


def write_compatible_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
    *,
    require_existing_match: bool,
) -> None:
    """Write a manifest without relabeling artifacts from another run.

    Formal resumes may reuse an experiment directory only when its complete
    manifest is identical.
    """
    destination = Path(path)
    payload = dict(manifest)
    if destination.exists() and require_existing_match:
        try:
            existing = json.loads(destination.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"existing run manifest is unreadable: {destination}"
            ) from error
        if not isinstance(existing, dict):
            raise ValueError(
                f"existing run manifest must be a JSON object: {destination}"
            )
        if existing != payload:
            keys = sorted(set(existing) | set(payload))
            differences = {
                key: {"existing": existing.get(key), "requested": payload.get(key)}
                for key in keys
                if existing.get(key) != payload.get(key)
            }
            preview = dict(list(differences.items())[:8])
            raise ValueError(
                "experiment directory belongs to a different run; "
                f"use a new directory. Differences: {preview}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def classify_run_for_resume(root: str | Path) -> ResumeState:
    """Classify an identity-matched run before launching another session."""
    run_root = Path(root)

    def load_object(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    parity = load_object(run_root / "parity.json")
    attempts = parity.get("attempts")
    failed_parity = parity.get("gate_passed") is False or (
        isinstance(attempts, list)
        and any(
            isinstance(attempt, dict) and attempt.get("gate_passed") is False
            for attempt in attempts
        )
    )
    if failed_parity:
        return "failed-parity"

    gpu = load_object(run_root / "gpu_metrics" / "gpu_summary.json")
    expected_devices = gpu.get("expected_device_count")
    if gpu.get("sampling_errors") or (
        expected_devices is not None
        and expected_devices != gpu.get("gpu_count")
    ):
        return "tainted-gpu-metrics"

    from vagen.analysis.analyze_rollouts import build_result_row

    return (
        "complete"
        if build_result_row(run_root)["Status"] == "complete"
        else "resumable"
    )
