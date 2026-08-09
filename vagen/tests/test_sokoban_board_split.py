"""Sokoban task identity is the generated board, not the requested seed.

``PatchedSokobanEnv.reset`` advances the requested seed until the generated room
falls inside ``min_solution_steps``, so two different requested seeds can yield
the same board.  A held-out split therefore has to be verified at the board
level; disjoint seed ranges alone do not establish that test puzzles are unseen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from vagen.analysis.experiment_contract import validate_experiment_contract
from vagen.analysis.sokoban_board_split import (
    check_split_consistency,
    load_split,
    select_board_disjoint_seeds,
)


ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = ROOT / "experiments" / "sokoban_board_split.json"
EVAL_CONFIG = ROOT / "examples" / "evaluate" / "sokoban" / "config.yaml"
TRAIN_CONFIG = ROOT / "examples" / "train" / "sokoban" / "train_sokoban_vision.yaml"
VAL_CONFIG = ROOT / "examples" / "train" / "sokoban" / "val_sokoban_vision.yaml"


def test_committed_split_is_internally_consistent() -> None:
    assert check_split_consistency(load_split(SPLIT_PATH)) == []


def test_test_boards_are_absent_from_train_and_validation() -> None:
    split = load_split(SPLIT_PATH)
    train = set(split["train"]["fingerprints"])
    validation = set(split["validation"]["fingerprints"].values())
    test = set(split["test"]["fingerprints"].values())

    assert validation, "the split records no validation boards"
    assert test, "the split records no test boards"
    assert not validation & train
    assert not test & train
    assert not test & validation
    assert len(validation) == split["validation"]["seed_count"]
    assert len(test) == split["test"]["seed_count"]


def test_requested_seeds_collapse_onto_fewer_boards() -> None:
    """Records the reason a seed-range split is not sufficient."""
    split = load_split(SPLIT_PATH)
    seed_count = split["train"]["seed_count"]
    board_count = split["train"]["unique_boards"]
    assert board_count < seed_count


@pytest.mark.parametrize(
    ("config_path", "split_role"),
    [(VAL_CONFIG, "validation"), (EVAL_CONFIG, "test")],
)
def test_runtime_config_uses_the_committed_seed_list(
    config_path: Path, split_role: str
) -> None:
    split = load_split(SPLIT_PATH)
    config = yaml.safe_load(config_path.read_text())
    environment = config["envs"][0]
    assert environment["seed_list"] == split[split_role]["seeds"]
    assert environment["n_envs"] == len(split[split_role]["seeds"])


def test_evaluation_matches_the_training_difficulty_window() -> None:
    """Test boards must differ from training boards in identity, not difficulty."""
    from vagen.envs.sokoban.sokoban_env import SokobanEnvConfig

    evaluation = yaml.safe_load(EVAL_CONFIG.read_text())["envs"][0]["config"]
    train = yaml.safe_load(TRAIN_CONFIG.read_text())["envs"][0]["config"]
    validation = yaml.safe_load(VAL_CONFIG.read_text())["envs"][0]["config"]

    assert evaluation["min_solution_steps"] == train["min_solution_steps"]
    assert evaluation["min_solution_steps"] == validation["min_solution_steps"]

    # The train/validation configs inherit room geometry from the dataclass
    # defaults, so the evaluation config must not silently diverge from them.
    defaults = SokobanEnvConfig()
    for key, default in (("dim_room", defaults.dim_room), ("num_boxes", defaults.num_boxes)):
        expected = train.get(key, default)
        assert list(evaluation[key]) == list(expected) if isinstance(
            expected, (list, tuple)
        ) else evaluation[key] == expected


def test_split_generation_config_matches_the_evaluated_config() -> None:
    split = load_split(SPLIT_PATH)
    generation = split["environment_config"]
    evaluation = yaml.safe_load(EVAL_CONFIG.read_text())["envs"][0]["config"]
    for key in ("dim_room", "num_boxes", "max_steps", "min_solution_steps"):
        assert list(generation[key]) == list(evaluation[key]) if isinstance(
            generation[key], list
        ) else generation[key] == evaluation[key]


def test_matrix_contract_requires_the_board_split() -> None:
    report = validate_experiment_contract(
        ROOT / "experiments" / "matrix.yaml", repo_root=ROOT
    )
    assert report["valid"] is True
    sokoban_evaluation = [
        entry
        for entry in report["checked_partitions"]
        if entry["environment"] == "sokoban" and entry["role"] == "evaluation"
    ]
    assert sokoban_evaluation
    assert sokoban_evaluation[0]["seed_selection"] == "explicit_list"


def test_contract_rejects_a_seed_list_that_drifts_from_the_split(tmp_path: Path) -> None:
    """Editing the eval config without regenerating the split must fail closed."""
    matrix = yaml.safe_load((ROOT / "experiments" / "matrix.yaml").read_text())
    split = json.loads(SPLIT_PATH.read_text())
    evaluation = yaml.safe_load(EVAL_CONFIG.read_text())

    evaluation["envs"][0]["seed_list"] = evaluation["envs"][0]["seed_list"][:-1] + [999999]

    for relative in (
        "experiments",
        "examples/evaluate/sokoban",
        "examples/train/sokoban",
        "examples/train/navigation",
        "examples/evaluate/navigation",
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    (tmp_path / "experiments" / "sokoban_board_split.json").write_text(json.dumps(split))
    (tmp_path / "examples/evaluate/sokoban/config.yaml").write_text(yaml.safe_dump(evaluation))
    for relative in (
        "examples/train/sokoban/train_sokoban_vision.yaml",
        "examples/train/sokoban/val_sokoban_vision.yaml",
        "examples/train/navigation/train_navigation.yaml",
        "examples/train/navigation/val_navigation.yaml",
        "examples/evaluate/navigation/config_base.yaml",
    ):
        (tmp_path / relative).write_text((ROOT / relative).read_text())
    matrix_path = tmp_path / "experiments" / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix))

    with pytest.raises(ValueError, match="does not match"):
        validate_experiment_contract(matrix_path, repo_root=tmp_path)


def test_contract_rejects_a_validation_list_that_drifts_from_the_split(
    tmp_path: Path,
) -> None:
    matrix = yaml.safe_load((ROOT / "experiments" / "matrix.yaml").read_text())
    split = json.loads(SPLIT_PATH.read_text())
    validation = yaml.safe_load(VAL_CONFIG.read_text())
    validation["envs"][0]["seed_list"][-1] = 999999

    for relative in (
        "experiments",
        "examples/evaluate/sokoban",
        "examples/train/sokoban",
        "examples/train/navigation",
        "examples/evaluate/navigation",
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    (tmp_path / "experiments" / "sokoban_board_split.json").write_text(
        json.dumps(split)
    )
    (tmp_path / "examples/train/sokoban/val_sokoban_vision.yaml").write_text(
        yaml.safe_dump(validation)
    )
    for relative in (
        "examples/evaluate/sokoban/config.yaml",
        "examples/train/sokoban/train_sokoban_vision.yaml",
        "examples/train/navigation/train_navigation.yaml",
        "examples/train/navigation/val_navigation.yaml",
        "examples/evaluate/navigation/config_base.yaml",
    ):
        (tmp_path / relative).write_text((ROOT / relative).read_text())
    matrix_path = tmp_path / "experiments" / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix))

    with pytest.raises(ValueError, match="validation seed_list does not match"):
        validate_experiment_contract(matrix_path, repo_root=tmp_path)


def test_selection_rejects_an_exhausted_candidate_stream() -> None:
    with pytest.raises(ValueError, match="board-disjoint"):
        select_board_disjoint_seeds(
            iter([]),
            excluded_fingerprints=set(),
            count=4,
            environment_config={
                "dim_room": [6, 6],
                "max_steps": 100,
                "num_boxes": 1,
                "min_solution_steps": [1, 5],
                "reset_seed_max_tries": 10,
                "min_solution_bfs_max_depth": 200,
            },
        )
