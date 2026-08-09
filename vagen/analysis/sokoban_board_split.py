"""Board-level train/validation/test partitioning for Sokoban.

Sokoban rooms are generated from a requested seed, and ``PatchedSokobanEnv.reset``
advances that seed until the generated room satisfies the configured
``min_solution_steps`` window.  Two different requested seeds can therefore
produce the same board, and a split that is only disjoint in *requested seeds*
can still evaluate on rooms the policy trained on.

This module fingerprints the board a requested seed actually produces, and
builds a test split whose boards are disjoint from train and validation under
the same generation config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


DIGEST_BYTES = 8

DEFAULT_ENVIRONMENT_CONFIG: dict[str, Any] = {
    "dim_room": [6, 6],
    "max_steps": 100,
    "num_boxes": 1,
    "min_solution_steps": [1, 5],
    "reset_seed_max_tries": 10000,
    "min_solution_bfs_max_depth": 200,
}


def _as_pair(value: Sequence[int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    pair = tuple(int(item) for item in value)
    if len(pair) != 2:
        raise ValueError(f"expected a two-element range, got {value!r}")
    return pair  # type: ignore[return-value]


def board_fingerprint(seed: int, environment_config: Mapping[str, Any]) -> str:
    """Return a stable digest of the board a requested seed actually produces.

    The digest covers both ``room_fixed`` (walls and targets) and ``room_state``
    (box and player placement), so it identifies the puzzle instance rather than
    the requested seed.
    """
    from vagen.envs.sokoban.patch_sokoban_env import PatchedSokobanEnv

    dim_room = tuple(int(value) for value in environment_config["dim_room"])
    env = PatchedSokobanEnv(
        dim_room=dim_room,
        max_steps=int(environment_config["max_steps"]),
        num_boxes=int(environment_config["num_boxes"]),
    )
    env.reset(
        render_mode="rgb_array",
        seed=int(seed),
        min_solution_steps=_as_pair(environment_config.get("min_solution_steps")),
        reset_seed_max_tries=int(environment_config["reset_seed_max_tries"]),
        min_solution_bfs_max_depth=int(environment_config["min_solution_bfs_max_depth"]),
    )
    payload = env.room_fixed.tobytes() + b"|" + env.room_state.tobytes()
    return hashlib.blake2s(payload, digest_size=DIGEST_BYTES).hexdigest()


def fingerprint_seeds(
    seeds: Iterable[int],
    environment_config: Mapping[str, Any],
) -> dict[int, str]:
    """Fingerprint every requested seed in order."""
    return {int(seed): board_fingerprint(seed, environment_config) for seed in seeds}


def select_board_disjoint_seeds(
    candidates: Iterator[int],
    excluded_fingerprints: Iterable[str],
    count: int,
    environment_config: Mapping[str, Any],
    *,
    max_candidates: int = 200_000,
) -> tuple[list[int], dict[int, str]]:
    """Pick ``count`` requested seeds with pairwise distinct, unseen boards.

    Args:
        candidates: Requested seeds to consider, in priority order.
        excluded_fingerprints: Boards already used by train or validation.
        count: Number of test seeds to select.
        environment_config: Generation config; must match the split being built.
        max_candidates: Safety bound on how many candidates are examined.

    Raises:
        ValueError: If the candidate stream is exhausted before ``count`` seeds
            with distinct unseen boards are found.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    blocked = set(excluded_fingerprints)
    chosen: list[int] = []
    chosen_fingerprints: dict[int, str] = {}
    examined = 0
    for seed in candidates:
        if len(chosen) >= count:
            break
        examined += 1
        if examined > max_candidates:
            break
        fingerprint = board_fingerprint(seed, environment_config)
        if fingerprint in blocked:
            continue
        blocked.add(fingerprint)
        chosen.append(int(seed))
        chosen_fingerprints[int(seed)] = fingerprint
    if len(chosen) < count:
        raise ValueError(
            f"only found {len(chosen)} board-disjoint seeds after {examined} candidates; "
            f"needed {count}"
        )
    return chosen, chosen_fingerprints


def build_split(
    *,
    train_range: tuple[int, int],
    validation_range: tuple[int, int],
    candidate_start: int,
    test_count: int,
    environment_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fingerprint train and validation, then pick a board-disjoint test split."""
    config = dict(environment_config or DEFAULT_ENVIRONMENT_CONFIG)
    train_seeds = range(train_range[0], train_range[1] + 1)
    validation_seeds = range(validation_range[0], validation_range[1] + 1)

    train_fingerprints = fingerprint_seeds(train_seeds, config)
    validation_fingerprints = fingerprint_seeds(validation_seeds, config)
    excluded = set(train_fingerprints.values()) | set(validation_fingerprints.values())

    candidates = iter(range(candidate_start, candidate_start + 200_000))
    test_seeds, test_fingerprints = select_board_disjoint_seeds(
        candidates, excluded, test_count, config
    )

    return {
        "schema_version": 1,
        "digest_bytes": DIGEST_BYTES,
        "environment_config": config,
        "train": {
            "seed_range": list(train_range),
            "seed_count": len(train_fingerprints),
            "unique_boards": len(set(train_fingerprints.values())),
            "fingerprints": sorted(set(train_fingerprints.values())),
        },
        "validation": {
            "seed_range": list(validation_range),
            "seed_count": len(validation_fingerprints),
            "unique_boards": len(set(validation_fingerprints.values())),
            "fingerprints": sorted(set(validation_fingerprints.values())),
        },
        "test": {
            "candidate_start": int(candidate_start),
            "seeds": test_seeds,
            "seed_count": len(test_seeds),
            "unique_boards": len(set(test_fingerprints.values())),
            "fingerprints": {str(seed): value for seed, value in test_fingerprints.items()},
        },
    }


def load_split(path: str | Path) -> dict[str, Any]:
    """Read a committed split artifact."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"split artifact must be a JSON object: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported split schema_version in {path}")
    return payload


def check_split_consistency(split: Mapping[str, Any]) -> list[str]:
    """Return structural problems in a split artifact without regenerating boards.

    The guarantee this checks is that the *test* boards are unseen: they must be
    pairwise distinct and absent from both train and validation.  Train and
    validation may legitimately share boards — validation only selects
    checkpoints, it makes no held-out claim — so their overlap is not an error.
    """
    problems: list[str] = []
    train = set(split["train"]["fingerprints"])
    validation = set(split["validation"]["fingerprints"])
    test_map = dict(split["test"]["fingerprints"])
    test = set(test_map.values())

    if len(test) != len(test_map):
        problems.append("test seeds map onto duplicate boards")
    overlap_train = test & train
    if overlap_train:
        problems.append(f"{len(overlap_train)} test boards also appear in train")
    overlap_validation = test & validation
    if overlap_validation:
        problems.append(f"{len(overlap_validation)} test boards also appear in validation")
    if len(test_map) != int(split["test"]["seed_count"]):
        problems.append("test seed_count does not match the fingerprint table")

    train_range = split["train"]["seed_range"]
    validation_range = split["validation"]["seed_range"]
    for seed in (int(value) for value in test_map):
        if train_range[0] <= seed <= train_range[1]:
            problems.append(f"test seed {seed} falls inside the train seed range")
        if validation_range[0] <= seed <= validation_range[1]:
            problems.append(f"test seed {seed} falls inside the validation seed range")
    return problems


def recompute_sample(
    split: Mapping[str, Any],
    sample_size: int,
) -> list[str]:
    """Regenerate a few test boards and confirm the committed digests still hold."""
    problems: list[str] = []
    config = split["environment_config"]
    entries = sorted(split["test"]["fingerprints"].items(), key=lambda item: int(item[0]))
    step = max(1, len(entries) // max(1, sample_size))
    for seed_text, expected in entries[::step][:sample_size]:
        actual = board_fingerprint(int(seed_text), config)
        if actual != expected:
            problems.append(
                f"seed {seed_text} now produces board {actual}, committed {expected}"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="fingerprint splits and pick test seeds")
    build_parser.add_argument("--train-range", type=int, nargs=2, default=(1, 10000))
    build_parser.add_argument("--validation-range", type=int, nargs=2, default=(10001, 10128))
    build_parser.add_argument("--candidate-start", type=int, default=20001)
    build_parser.add_argument("--test-count", type=int, default=128)
    build_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify", help="check a committed split artifact")
    verify_parser.add_argument("--split", type=Path, required=True)
    verify_parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="regenerate this many test boards to confirm the digests",
    )

    args = parser.parse_args()

    if args.command == "build":
        split = build_split(
            train_range=tuple(args.train_range),
            validation_range=tuple(args.validation_range),
            candidate_start=args.candidate_start,
            test_count=args.test_count,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "train_unique_boards": split["train"]["unique_boards"],
                    "validation_unique_boards": split["validation"]["unique_boards"],
                    "test_seeds": split["test"]["seed_count"],
                    "test_unique_boards": split["test"]["unique_boards"],
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    split = load_split(args.split)
    problems = check_split_consistency(split)
    if args.sample:
        problems.extend(recompute_sample(split, args.sample))
    print(json.dumps({"valid": not problems, "problems": problems}, indent=2, sort_keys=True))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
