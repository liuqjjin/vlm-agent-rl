"""Runtime-backed train/validation/test split contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vagen.evaluate.run_eval import _load_config, _parse_env_specs
from vagen.gym_agent_dataset import load_envspecs


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("environment", ["sokoban", "navigation"])
def test_matrix_split_declarations_match_runtime_configs(environment: str) -> None:
    matrix = OmegaConf.load(ROOT / "experiments/matrix.yaml")
    declaration = matrix.environments[environment]

    train = load_envspecs(str(ROOT / declaration.train)).specs[0]
    validation = load_envspecs(str(ROOT / declaration.validation)).specs[0]
    evaluation_config = OmegaConf.to_container(
        _load_config(str(ROOT / declaration.evaluation), []), resolve=True
    )
    evaluation = _parse_env_specs(evaluation_config)[0]

    assert list(train.seed[:2]) == list(declaration.train_seeds)
    assert list(validation.seed[:2]) == list(declaration.validation_seeds)
    if evaluation.seed_list:
        # Environments whose requested seed is not a task identity enumerate
        # their held-out tasks; the matrix then declares the enumerated bounds.
        evaluation_bounds = [min(evaluation.seed_list), max(evaluation.seed_list)]
    else:
        evaluation_bounds = list(evaluation.seed[:2])
    assert evaluation_bounds == list(declaration.evaluation_seeds)


def test_navigation_split_identity_includes_dataset_domain() -> None:
    """Numeric task ids may repeat across the distinct base/base_train sets."""
    matrix = OmegaConf.load(ROOT / "experiments/matrix.yaml")
    declaration = matrix.environments.navigation
    train = load_envspecs(str(ROOT / declaration.train)).specs[0]
    validation = load_envspecs(str(ROOT / declaration.validation)).specs[0]
    evaluation_config = OmegaConf.to_container(
        _load_config(str(ROOT / declaration.evaluation), []), resolve=True
    )
    evaluation = _parse_env_specs(evaluation_config)[0]

    assert train.config["eval_set"] == "base_train"
    assert validation.config["eval_set"] == "base"
    assert evaluation.config["eval_set"] == "base"
    validation_ids = set(
        range(
            int(declaration.validation_seeds[0]),
            int(declaration.validation_seeds[1]) + 1,
        )
    )
    evaluation_ids = set(
        range(
            int(declaration.evaluation_seeds[0]),
            int(declaration.evaluation_seeds[1]) + 1,
        )
    )
    assert validation_ids.isdisjoint(evaluation_ids)
