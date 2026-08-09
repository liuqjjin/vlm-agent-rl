"""Reported success rates must be reachable on their own episode counts.

The result table pools three training seeds per method (Sokoban 3x128, Navigation
3x30) and evaluates the base model once (128 / 30).  A success rate is therefore a
ratio of integers, and a mean turn count is a ratio of an integer turn sum to an
integer success count.  A published cell that no integer outcome can produce is a
transcription error, so the table is checked against its own denominators.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"

TRAIN_SEEDS = 3
# (environment, per-checkpoint episodes, per-episode turn cap)
ENVIRONMENTS = {
    "Sokoban": (128, 5),
    "Navigation": (30, 10),
}
CELL = re.compile(r"(\d+\.\d+)%（(\d+)/(\d+)）")


def _result_rows() -> list[list[str]]:
    lines = README.read_text().split("\n")
    header = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("| 方法 |") and "Sokoban 成功率" in line
    )
    rows = []
    for line in lines[header + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def test_result_table_has_one_row_per_method() -> None:
    rows = _result_rows()
    assert len(rows) == 4, rows
    assert rows[0][0].startswith("Base")


@pytest.mark.parametrize("column,environment", [(1, "Sokoban"), (3, "Navigation")])
def test_success_rates_are_reachable_on_their_denominators(
    column: int, environment: str
) -> None:
    episodes, _ = ENVIRONMENTS[environment]
    for row in _result_rows():
        match = CELL.search(row[column])
        assert match, f"{row[0]} / {environment} does not report successes/episodes"
        percent, successes, denominator = (
            float(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        expected = episodes if row[0].startswith("Base") else episodes * TRAIN_SEEDS
        assert denominator == expected, (
            f"{row[0]} / {environment}: denominator {denominator} != {expected}"
        )
        assert successes <= denominator
        assert round(100 * successes / denominator, 1) == percent, (
            f"{row[0]} / {environment}: {successes}/{denominator} is not {percent}%"
        )


@pytest.mark.parametrize(
    "rate_column,turn_column,environment",
    [(1, 2, "Sokoban"), (3, 4, "Navigation")],
)
def test_mean_turns_are_reachable_and_within_the_episode_cap(
    rate_column: int, turn_column: int, environment: str
) -> None:
    _, cap = ENVIRONMENTS[environment]
    for row in _result_rows():
        successes = int(CELL.search(row[rate_column]).group(2))
        mean = float(row[turn_column])
        assert 1.0 <= mean <= cap, f"{row[0]} / {environment}: mean {mean} outside [1, {cap}]"
        reachable = any(
            round(total / successes, 1) == mean
            for total in range(successes, cap * successes + 1)
        )
        assert reachable, (
            f"{row[0]} / {environment}: mean {mean} unreachable over {successes} successes"
        )


def test_episode_counts_match_the_evaluation_configs() -> None:
    sokoban = yaml.safe_load(
        (ROOT / "examples/evaluate/sokoban/config.yaml").read_text()
    )["envs"][0]
    navigation = yaml.safe_load(
        (ROOT / "examples/evaluate/navigation/config_base.yaml").read_text()
    )["envs"][0]
    assert sokoban["n_envs"] == ENVIRONMENTS["Sokoban"][0]
    assert navigation["n_envs"] == ENVIRONMENTS["Navigation"][0]
    assert sokoban["max_turns"] == ENVIRONMENTS["Sokoban"][1]
    assert navigation["max_turns"] == ENVIRONMENTS["Navigation"][1]


def test_resume_reports_the_same_headline_numbers() -> None:
    """The resume must not drift from the README table it summarizes."""
    resume = (ROOT / "RESUME_PROJECT_CN.md").read_text()
    rows = {row[0]: row for row in _result_rows()}
    episode_row = next(row for name, row in rows.items() if "episode GRPO" in name)
    base_row = rows[next(name for name in rows if name.startswith("Base"))]

    for column in (1, 3):
        for row in (episode_row, base_row):
            percent = CELL.search(row[column]).group(1)
            assert f"{percent}%" in resume, f"{percent}% missing from the resume"
