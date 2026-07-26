from __future__ import annotations

import pytest

from scripts.run_with_gpu_metrics import (
    parse_nvidia_smi_rows,
    parse_visible_devices,
    merge_session_samples,
    summarize_samples,
    validate_sampled_device_count,
)


def test_gpu_metrics_parser_and_summary():
    first = parse_nvidia_smi_rows(
        "0, NVIDIA A100-SXM4-80GB, 1024, 81920, 50, 200\n"
        "1, NVIDIA A100-SXM4-80GB, 2048, 81920, 75, 250\n",
        elapsed_seconds=0.0,
    )
    second = parse_nvidia_smi_rows(
        "0, NVIDIA A100-SXM4-80GB, 4096, 81920, 90, 300\n"
        "1, NVIDIA A100-SXM4-80GB, 3072, 81920, 80, 350\n",
        elapsed_seconds=3600.0,
    )
    summary = summarize_samples(first + second, duration_seconds=3600.0)
    assert summary["gpu_count"] == 2
    assert summary["gpu_hours"] == pytest.approx(2.0)
    assert summary["peak_vram_mib"] == pytest.approx(4096.0)
    assert summary["peak_vram_mib_by_device"] == {
        "0": 4096.0,
        "1": 3072.0,
    }
    assert summary["energy_kwh_estimate"] == pytest.approx(0.55)
    assert summary["mean_utilization_percent_by_device"] == {
        "0": pytest.approx(70.0),
        "1": pytest.approx(77.5),
    }


def test_gpu_metrics_parser_ignores_malformed_rows():
    assert parse_nvidia_smi_rows("not supported\n", 1.0) == []
    summary = summarize_samples([], duration_seconds=10.0)
    assert summary["gpu_count"] == 0
    assert summary["gpu_hours"] is None
    assert summary["peak_vram_mib"] is None
    assert summary["energy_kwh_estimate"] is None


def test_gpu_metrics_keeps_memory_when_power_is_unavailable():
    samples = parse_nvidia_smi_rows(
        "0, NVIDIA A100-SXM4-80GB, 4096, 81920, 90, [N/A]\n",
        elapsed_seconds=1.0,
    )
    assert samples[0]["memory_used_mib"] == pytest.approx(4096.0)
    assert samples[0]["power_watts"] is None
    summary = summarize_samples(samples, duration_seconds=2.0)
    assert summary["peak_vram_mib"] == pytest.approx(4096.0)
    assert summary["energy_kwh_estimate"] is None


def test_cuda_visible_devices_are_parsed_without_sampling_unallocated_gpus():
    assert parse_visible_devices(None) is None
    assert parse_visible_devices("all") is None
    assert parse_visible_devices("") == []
    assert parse_visible_devices("-1") == []
    assert parse_visible_devices("2, GPU-abc") == ["2", "GPU-abc"]
    with pytest.raises(ValueError, match="duplicates"):
        parse_visible_devices("0,0")


def test_declared_gpu_count_must_match_sampled_devices():
    samples = parse_nvidia_smi_rows(
        "0, NVIDIA A100, 100, 1000, 10, 50\n"
        "1, NVIDIA A100, 200, 1000, 20, 60\n",
        elapsed_seconds=0.0,
    )
    validate_sampled_device_count(samples, 2)
    with pytest.raises(RuntimeError, match="sampled 2 devices"):
        validate_sampled_device_count(samples, 1)


def test_resumed_gpu_samples_accumulate_without_charging_offline_gap():
    prior = parse_nvidia_smi_rows(
        "0, NVIDIA A100, 100, 1000, 10, 100\n",
        elapsed_seconds=10.0,
    )
    current = parse_nvidia_smi_rows(
        "0, NVIDIA A100, 200, 1000, 20, 200\n",
        elapsed_seconds=0.0,
    )
    merged = merge_session_samples(
        prior,
        current,
        prior_duration_seconds=10.0,
    )
    assert [row["elapsed_seconds"] for row in merged] == [10.0, 10.0]
    summary = summarize_samples(merged, duration_seconds=20.0)
    assert summary["gpu_hours"] == pytest.approx(20.0 / 3600.0)
    assert summary["energy_kwh_estimate"] == pytest.approx(0.0)

    changed_device = parse_nvidia_smi_rows(
        "0, NVIDIA H100, 200, 80000, 20, 200\n",
        elapsed_seconds=0.0,
    )
    with pytest.raises(RuntimeError, match="different device inventories"):
        merge_session_samples(
            prior,
            changed_device,
            prior_duration_seconds=10.0,
        )
