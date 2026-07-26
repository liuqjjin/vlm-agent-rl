from __future__ import annotations

import pytest

from scripts.run_with_gpu_metrics import parse_nvidia_smi_rows, summarize_samples


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


def test_gpu_metrics_parser_ignores_malformed_rows():
    assert parse_nvidia_smi_rows("not supported\n", 1.0) == []
    summary = summarize_samples([], duration_seconds=10.0)
    assert summary["gpu_count"] == 0
    assert summary["peak_vram_mib"] is None
    assert summary["energy_kwh_estimate"] is None
