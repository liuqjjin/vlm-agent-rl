from __future__ import annotations

import csv
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vagen.evaluate.run_eval import _load_config, _parse_env_specs
from vagen.evaluate.utils.seeding_utils import generate_seeds_for_spec
from vagen.gym_agent_dataset import (
    EnvSpec,
    _generate_seeds_for_spec,
    load_envspecs,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    [
        (
            "examples/evaluate/frozenlake/config.yaml",
            set(range(10001, 10129)),
        ),
        (
            "examples/evaluate/sokoban/config.yaml",
            set(range(10129, 10257)),  # Updated to [10129, 10256]
        ),
        (
            "examples/evaluate/navigation/config_base.yaml",
            set(range(30, 60)),  # Updated to [30, 59]
        ),
    ],
)
def test_formal_evaluation_configs_cover_exact_seed_sets(
    relative_path: str, expected: set[int]
) -> None:
    path = ROOT / relative_path
    config = OmegaConf.to_container(
        _load_config(str(path), []), resolve=True
    )
    spec = _parse_env_specs(config)[0]
    seeds = generate_seeds_for_spec(spec, base_seed=0, spec_idx=0)
    assert len(seeds) == len(expected)
    assert set(seeds) == expected


def test_navigation_training_validation_and_test_task_domains_are_disjoint() -> None:
    train = load_envspecs(
        str(ROOT / "examples/train/navigation/train_navigation.yaml")
    ).specs[0]
    validation = load_envspecs(
        str(ROOT / "examples/train/navigation/val_navigation.yaml")
    ).specs[0]
    evaluation_config = OmegaConf.to_container(
        _load_config(
            str(ROOT / "examples/evaluate/navigation/config_base.yaml"), []
        ),
        resolve=True,
    )
    evaluation = _parse_env_specs(evaluation_config)[0]

    train_seeds = _generate_seeds_for_spec(train, base_seed=0, spec_idx=0)
    validation_seeds = _generate_seeds_for_spec(validation, base_seed=0, spec_idx=0)
    evaluation_seeds = generate_seeds_for_spec(evaluation, base_seed=0, spec_idx=0)

    assert train.config["eval_set"] == "base_train"
    assert validation.config["eval_set"] == "base"
    assert evaluation.config["eval_set"] == "base"
    assert set(train_seeds) == set(range(1200))
    assert set(validation_seeds) == set(range(30))
    assert set(evaluation_seeds) == set(range(30, 60))

    train_tasks = {(train.config["eval_set"], seed) for seed in train_seeds}
    validation_tasks = {
        (validation.config["eval_set"], seed) for seed in validation_seeds
    }
    evaluation_tasks = {
        (evaluation.config["eval_set"], seed) for seed in evaluation_seeds
    }
    assert train_tasks.isdisjoint(validation_tasks)
    assert train_tasks.isdisjoint(evaluation_tasks)
    assert validation_tasks.isdisjoint(evaluation_tasks)


@pytest.mark.parametrize("environment", ["frozenlake", "sokoban", "navigation"])
def test_formal_evaluation_seeds_are_held_out_from_training(
    environment: str,
) -> None:
    train = load_envspecs(
        str(ROOT / f"examples/train/{environment}/train_{environment}_vision.yaml")
        if environment != "navigation"
        else str(ROOT / "examples/train/navigation/train_navigation.yaml")
    ).specs[0]
    evaluation_path = (
        ROOT / "examples/evaluate/navigation/config_base.yaml"
        if environment == "navigation"
        else ROOT / f"examples/evaluate/{environment}/config.yaml"
    )
    evaluation_config = OmegaConf.to_container(
        _load_config(str(evaluation_path), []), resolve=True
    )
    evaluation = _parse_env_specs(evaluation_config)[0]

    train_domain = train.config.get("eval_set", environment)
    evaluation_domain = evaluation.config.get("eval_set", environment)
    train_tasks = {
        (train_domain, seed)
        for seed in _generate_seeds_for_spec(train, base_seed=0, spec_idx=0)
    }
    evaluation_tasks = {
        (evaluation_domain, seed)
        for seed in generate_seeds_for_spec(evaluation, base_seed=0, spec_idx=0)
    }
    assert train_tasks.isdisjoint(evaluation_tasks)


def test_training_seed_list_accepts_exact_declared_count() -> None:
    spec = EnvSpec(name="test", n_envs=3, seed_list=[7, 8, 9])
    assert _generate_seeds_for_spec(spec, base_seed=0, spec_idx=0) == [7, 8, 9]


def test_pending_result_registry_matches_the_experiment_matrix() -> None:
    matrix = OmegaConf.load(ROOT / "experiments/matrix.yaml")
    with (ROOT / "results/main_results.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    expected_columns = {
        "Method",
        "Visual Success",
        "Peak VRAM",
        "GPU·h",
        "Mean Turns",
        "Ratio P95",
        "Status",
        "Environment",
        "Seed",
        "Commit",
        "Evidence",
    }
    assert set(reader.fieldnames or []) == expected_columns
    base_rows = {
        str(row["Environment"]).lower(): row
        for row in rows
        if row["Method"] == "Base Qwen2.5-VL-3B"
    }
    for environment in ("sokoban", "navigation"):
        declared = list(matrix.environments[environment].evaluation_seeds)
        row = base_rows[environment]
        assert row["Seed"] == f"{declared[0]}:{declared[1]}"

    for row in rows:
        assert row["Status"] == "pending-external-gpu"
        assert row["Commit"] == ""
        for metric in (
            "Visual Success",
            "Peak VRAM",
            "GPU·h",
            "Mean Turns",
            "Ratio P95",
        ):
            assert row[metric] == ""
