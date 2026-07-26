#!/usr/bin/env python3
"""Run a command while recording peak VRAM, utilization, energy, and GPU-hours."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


QUERY_FIELDS = (
    "index",
    "name",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "power.draw",
)


def parse_visible_devices(value: str | None) -> list[str] | None:
    """Translate CUDA visibility into selectors accepted by ``nvidia-smi``.

    ``None`` means the variable is unset and all host devices are intentionally
    sampled. An empty list means CUDA explicitly hides every device.
    """
    if value is None or value.strip().lower() == "all":
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"-1", "none", "void"}:
        return []
    selectors = [item.strip() for item in normalized.split(",") if item.strip()]
    if len(selectors) != len(set(selectors)):
        raise ValueError(f"CUDA_VISIBLE_DEVICES contains duplicates: {value!r}")
    return selectors


def _optional_float(value: str) -> float | None:
    return (
        None
        if value.strip().lower() in {"", "n/a", "[n/a]", "not supported"}
        else float(value)
    )


def parse_nvidia_smi_rows(output: str, elapsed_seconds: float) -> list[dict[str, Any]]:
    """Parse numeric no-unit nvidia-smi rows into timestamped samples."""
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != len(QUERY_FIELDS):
            continue
        try:
            rows.append(
                {
                    "elapsed_seconds": elapsed_seconds,
                    "gpu_index": int(fields[0]),
                    "gpu_name": fields[1],
                    "memory_used_mib": float(fields[2]),
                    "memory_total_mib": float(fields[3]),
                    "utilization_percent": _optional_float(fields[4]),
                    "power_watts": _optional_float(fields[5]),
                }
            )
        except ValueError:
            continue
    return rows


def sample_gpus(
    elapsed_seconds: float,
    device_selectors: list[str] | None = None,
) -> list[dict[str, Any]]:
    if device_selectors == []:
        return []
    command = ["nvidia-smi"]
    if device_selectors is not None:
        command.append(f"--id={','.join(device_selectors)}")
    command.extend(
        [
            f"--query-gpu={','.join(QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nvidia_smi_rows(result.stdout, elapsed_seconds)


def summarize_samples(
    samples: list[dict[str, Any]],
    *,
    duration_seconds: float,
) -> dict[str, Any]:
    """Aggregate device samples without claiming metrics when none exist."""
    devices = sorted({int(row["gpu_index"]) for row in samples})
    peaks = {
        str(device): max(
            row["memory_used_mib"]
            for row in samples
            if int(row["gpu_index"]) == device
        )
        for device in devices
    }
    names = {
        str(device): next(
            str(row["gpu_name"])
            for row in samples
            if int(row["gpu_index"]) == device
        )
        for device in devices
    }
    mean_utilization = {}
    for device in devices:
        values = [
            float(row["utilization_percent"])
            for row in samples
            if int(row["gpu_index"]) == device
            and row["utilization_percent"] is not None
        ]
        mean_utilization[str(device)] = (
            sum(values) / len(values) if values else None
        )
    energy_wh = 0.0
    has_power_samples = False
    for device in devices:
        device_rows = sorted(
            (
                row
                for row in samples
                if int(row["gpu_index"]) == device
            ),
            key=lambda row: row["elapsed_seconds"],
        )
        for left, right in zip(device_rows, device_rows[1:]):
            if left["power_watts"] is None or right["power_watts"] is None:
                continue
            has_power_samples = True
            delta_hours = (
                right["elapsed_seconds"] - left["elapsed_seconds"]
            ) / 3600.0
            mean_power = (left["power_watts"] + right["power_watts"]) / 2.0
            energy_wh += mean_power * delta_hours

    duration_hours = duration_seconds / 3600.0
    return {
        "duration_seconds": duration_seconds,
        "gpu_count": len(devices),
        "gpu_hours": duration_hours * len(devices) if devices else None,
        "peak_vram_mib": max(peaks.values(), default=None),
        "peak_vram_mib_by_device": peaks,
        "gpu_name_by_device": names,
        "mean_utilization_percent_by_device": mean_utilization,
        "energy_kwh_estimate": (
            energy_wh / 1000.0 if has_power_samples else None
        ),
        "sample_count": len(samples),
    }


def _write_samples(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "elapsed_seconds",
        "gpu_index",
        "gpu_name",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_percent",
        "power_watts",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.interval <= 0:
        parser.error("--interval must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visible_devices_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_selectors = parse_visible_devices(visible_devices_raw)
    start = time.monotonic()
    process = subprocess.Popen(command)
    samples: list[dict[str, Any]] = []
    sample_errors: list[str] = []

    while process.poll() is None:
        elapsed = time.monotonic() - start
        try:
            samples.extend(sample_gpus(elapsed, device_selectors))
        except Exception as error:
            sample_errors.append(repr(error))
        try:
            process.wait(timeout=args.interval)
        except subprocess.TimeoutExpired:
            pass

    duration = time.monotonic() - start
    try:
        samples.extend(sample_gpus(duration, device_selectors))
    except Exception as error:
        sample_errors.append(repr(error))

    _write_samples(args.output_dir / "gpu_samples.csv", samples)
    summary = summarize_samples(samples, duration_seconds=duration)
    summary.update(
        {
            "command": command,
            "return_code": process.returncode,
            "sampling_interval_seconds": args.interval,
            "sampling_errors": sample_errors,
            "cuda_visible_devices": visible_devices_raw,
            "sampled_device_selectors": device_selectors,
        }
    )
    (args.output_dir / "gpu_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        "[GPU METRICS] "
        f"peak_vram_mib={summary['peak_vram_mib']} "
        f"gpu_hours={summary['gpu_hours']} "
        f"return_code={process.returncode}",
        file=sys.stderr,
    )
    return int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
