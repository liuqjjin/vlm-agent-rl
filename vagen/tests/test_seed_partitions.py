from __future__ import annotations

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
            set(range(10001, 10129)),
        ),
        (
            "examples/evaluate/navigation/config_base.yaml",
            set(range(30, 60)),
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


def test_navigation_training_and_validation_seed_domains_do_not_overlap() -> None:
    train = load_envspecs(
        str(ROOT / "examples/train/navigation/train_navigation.yaml")
    ).specs[0]
    validation = load_envspecs(
        str(ROOT / "examples/train/navigation/val_navigation.yaml")
    ).specs[0]
    train_seeds = _generate_seeds_for_spec(train, base_seed=0, spec_idx=0)
    validation_seeds = _generate_seeds_for_spec(
        validation, base_seed=0, spec_idx=0
    )

    assert len(train_seeds) == 10000
    assert set(train_seeds) <= set(range(0, 30))
    assert set(validation_seeds) == set(range(30, 60))
    assert set(train_seeds).isdisjoint(validation_seeds)


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

    train_seeds = set(
        _generate_seeds_for_spec(train, base_seed=0, spec_idx=0)
    )
    evaluation_seeds = set(
        generate_seeds_for_spec(evaluation, base_seed=0, spec_idx=0)
    )
    assert train_seeds.isdisjoint(evaluation_seeds)


def test_training_seed_list_accepts_exact_declared_count() -> None:
    spec = EnvSpec(name="test", n_envs=3, seed_list=[7, 8, 9])
    assert _generate_seeds_for_spec(spec, base_seed=0, spec_idx=0) == [7, 8, 9]
