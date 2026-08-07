"""Validate the declarative experiment matrix against executable YAML configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except OSError as error:
        raise ValueError(f"cannot read config: {path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _first_environment(path: Path) -> dict[str, Any]:
    config = _load_mapping(path)
    environments = config.get("envs")
    if not isinstance(environments, list) or len(environments) != 1:
        raise ValueError(f"{path} must define exactly one envs entry")
    environment = environments[0]
    if not isinstance(environment, dict):
        raise ValueError(f"{path} envs[0] must be a mapping")
    return environment


def _seed_range(value: Any, *, path: Path) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) not in {2, 3}:
        raise ValueError(f"{path} seed must be [start, end, repeat]")
    if not all(isinstance(item, int) for item in value):
        raise ValueError(f"{path} seed entries must be integers")
    start, end = value[:2]
    repeat = value[2] if len(value) == 3 else 0
    if start > end:
        raise ValueError(f"{path} seed start exceeds end")
    return start, end, repeat


def _identity(environment: dict[str, Any], seed: int) -> tuple[str, int]:
    config = environment.get("config")
    config = config if isinstance(config, dict) else {}
    task_set = config.get("eval_set")
    if task_set is None:
        task_set = environment.get("data_source", environment.get("name", "unknown"))
    return str(task_set), seed


def validate_experiment_contract(
    matrix_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Return a report and raise ``ValueError`` when matrix/configs disagree."""
    matrix_path = matrix_path.resolve()
    repo_root = (repo_root or matrix_path.parent.parent).resolve()
    matrix = _load_mapping(matrix_path)
    errors: list[str] = []
    checked: list[dict[str, Any]] = []

    if matrix.get("schema_version") != 1:
        errors.append("matrix schema_version must be 1")
    methods = matrix.get("methods")
    supported_methods = {
        "base",
        "concat_grpo",
        "no_concat_gae",
        "no_concat_episode_grpo",
    }
    if not isinstance(methods, dict) or set(methods) != supported_methods:
        errors.append(
            f"matrix methods must be exactly {sorted(supported_methods)}"
        )

    funnel = matrix.get("funnel")
    funnel = funnel if isinstance(funnel, dict) else {}
    confirmatory = funnel.get("confirmatory")
    confirmatory = confirmatory if isinstance(confirmatory, dict) else {}
    confirmatory_seeds = confirmatory.get("seeds")
    if confirmatory_seeds != [0, 1, 2]:
        errors.append("confirmatory seeds must be the independent training seeds [0, 1, 2]")
    if confirmatory.get("steps") != 401:
        errors.append("confirmatory steps must match the executable default (401)")

    environments = matrix.get("environments")
    if not isinstance(environments, dict) or not environments:
        errors.append("matrix environments must be a non-empty mapping")
        environments = {}
    confirmatory_environments = confirmatory.get("environments", [])
    if set(confirmatory_environments) != set(environments):
        errors.append("confirmatory environments must match matrix environments")

    path_keys = (
        ("train", "train_seeds"),
        ("validation", "validation_seeds"),
        ("evaluation", "evaluation_seeds"),
    )
    for environment_name, specification in environments.items():
        if not isinstance(specification, dict):
            errors.append(f"environment {environment_name} must be a mapping")
            continue
        identities: dict[str, set[tuple[str, int]]] = {}
        for role, seed_key in path_keys:
            relative_path = specification.get(role)
            declared_range = specification.get(seed_key)
            if not isinstance(relative_path, str):
                errors.append(f"{environment_name}.{role} path is missing")
                continue
            path = (repo_root / relative_path).resolve()
            try:
                path.relative_to(repo_root)
            except ValueError:
                errors.append(f"{environment_name}.{role} escapes repository root")
                continue
            if not path.is_file():
                errors.append(f"{environment_name}.{role} does not exist: {path}")
                continue
            try:
                environment = _first_environment(path)
                start, end, repeat = _seed_range(environment.get("seed"), path=path)
            except ValueError as error:
                errors.append(str(error))
                continue
            if declared_range != [start, end]:
                errors.append(
                    f"{environment_name}.{seed_key}={declared_range!r} does not match "
                    f"{relative_path} seed range [{start}, {end}]"
                )
            expected_count = end - start + 1
            n_envs = environment.get("n_envs")
            if n_envs != expected_count:
                errors.append(
                    f"{relative_path} n_envs={n_envs!r}, expected {expected_count}"
                )
            if role in {"train", "validation"} and repeat != 1:
                errors.append(
                    f"{relative_path} must set seed repeat=1 for unique task coverage"
                )
            if role == "evaluation" and repeat not in {0, 1}:
                errors.append(f"{relative_path} has invalid evaluation seed repeat={repeat}")
            identities[role] = {
                _identity(environment, seed) for seed in range(start, end + 1)
            }
            checked.append(
                {
                    "environment": environment_name,
                    "role": role,
                    "config": str(path),
                    "task_set": _identity(environment, start)[0],
                    "seed_start": start,
                    "seed_end": end,
                    "n_envs": n_envs,
                }
            )
        roles = list(identities)
        for index, left in enumerate(roles):
            for right in roles[index + 1 :]:
                overlap = identities[left] & identities[right]
                if overlap:
                    examples = sorted(overlap)[:3]
                    errors.append(
                        f"{environment_name} {left}/{right} task identities overlap: {examples}"
                    )

    result_columns = matrix.get("required_result_columns")
    required_columns = {
        "Method",
        "Visual Success",
        "Peak VRAM",
        "GPU·h",
        "Mean Turns",
        "Ratio P95",
    }
    if not isinstance(result_columns, list) or not required_columns.issubset(
        result_columns
    ):
        errors.append("required_result_columns is incomplete")

    report = {
        "schema_version": 1,
        "matrix": str(matrix_path),
        "valid": not errors,
        "errors": errors,
        "checked_partitions": checked,
    }
    if errors:
        raise ValueError("experiment contract validation failed:\n- " + "\n- ".join(errors))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate_experiment_contract(
            args.matrix, repo_root=args.repo_root
        )
    except ValueError as error:
        parser.exit(2, f"[ERROR] {error}\n")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
