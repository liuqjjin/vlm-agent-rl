"""Configuration validation utilities for entry points.

Validates training and evaluation configurations before expensive GPU runs,
catching common misconfigurations early.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


_VALID_ADVANTAGE_ESTIMATORS = {
    "gae",
    "gpg",
    "grpo",
    "grpo_passk",
    "grpo_vectorized",
    "no_concat_episode_grpo",
    "no_concat_gae",
    "no_concat_gae_last",
    "opo",
    "reinforce_plus_plus",
    "reinforce_plus_plus_baseline",
    "remax",
    "rloo",
    "rloo_vectorized",
}
_CRITIC_ESTIMATORS = {"gae", "no_concat_gae", "no_concat_gae_last"}
_GROUP_RELATIVE_ESTIMATORS = {
    "grpo",
    "grpo_passk",
    "grpo_vectorized",
    "no_concat_episode_grpo",
    "opo",
    "rloo",
    "rloo_vectorized",
}


def validate_training_config(config: dict[str, Any]) -> list[str]:
    """Validate training configuration and return list of issues.

    Args:
        config: Training configuration dictionary

    Returns:
        List of validation error messages (empty if valid)

    Example:
        >>> config = {"trainer": {"n_gpus": 4}, "actor_rollout_ref": {"actor": {"strategy": "fsdp"}}}
        >>> errors = validate_training_config(config)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"ERROR: {error}")
    """
    errors = []

    # Required top-level keys
    required_keys = ["trainer", "actor_rollout_ref", "algorithm"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required configuration key: {key}")

    if "trainer" in config:
        trainer = config["trainer"]

        # GPU configuration
        n_gpus = trainer.get("n_gpus_per_node", 0)
        nnodes = trainer.get("nnodes", 1)
        if n_gpus <= 0:
            errors.append(f"Invalid n_gpus_per_node: {n_gpus} (must be > 0)")
        if nnodes <= 0:
            errors.append(f"Invalid nnodes: {nnodes} (must be > 0)")

        # Training steps
        total_steps = trainer.get("total_training_steps")
        if total_steps is not None and total_steps <= 0:
            errors.append(
                f"Invalid total_training_steps: {total_steps} (must be > 0)"
            )

    # Actor rollout configuration
    if "actor_rollout_ref" in config:
        model_config = config["actor_rollout_ref"].get("model", {})
        actor_config = config["actor_rollout_ref"].get("actor", {})
        rollout_config = config["actor_rollout_ref"].get("rollout", {})

        # Strategy validation
        strategy = actor_config.get("strategy")
        if strategy not in {"fsdp", "fsdp2", "megatron", None}:
            errors.append(f"Unknown actor strategy: {strategy}")

        rollout_n = rollout_config.get("n", 1)
        if not isinstance(rollout_n, int) or isinstance(rollout_n, bool) or rollout_n <= 0:
            errors.append(f"Invalid rollout.n: {rollout_n} (must be a positive integer)")

        lora_rank = model_config.get("lora_rank", 0)
        if isinstance(lora_rank, int) and lora_rank > 0:
            if rollout_config.get("load_format") != "safetensors":
                errors.append(
                    "LoRA rollout requires rollout.load_format=safetensors"
                )
            if rollout_config.get("layered_summon") is not True:
                errors.append("LoRA rollout requires rollout.layered_summon=True")

    # Algorithm configuration
    if "algorithm" in config:
        algo = config["algorithm"]

        # Advantage estimator
        adv_estimator = algo.get("adv_estimator")
        if adv_estimator is None:
            errors.append("Missing required configuration key: algorithm.adv_estimator")
        elif adv_estimator not in _VALID_ADVANTAGE_ESTIMATORS:
            errors.append(
                f"Unknown algorithm.adv_estimator: {adv_estimator} "
                f"(valid: {sorted(_VALID_ADVANTAGE_ESTIMATORS)})"
            )

        # GAE parameters
        if adv_estimator in _CRITIC_ESTIMATORS:
            gamma = algo.get("gamma")
            if gamma is not None and not (0.0 <= gamma <= 1.0):
                errors.append(f"Invalid gamma: {gamma} (must be in [0, 1])")

            lam = algo.get("lam")
            if lam is not None and not (0.0 <= lam <= 1.0):
                errors.append(f"Invalid lambda: {lam} (must be in [0, 1])")

        rollout_n = config.get("actor_rollout_ref", {}).get("rollout", {}).get("n", 1)
        if (
            adv_estimator in _GROUP_RELATIVE_ESTIMATORS
            and isinstance(rollout_n, int)
            and not isinstance(rollout_n, bool)
            and rollout_n < 2
        ):
            errors.append(
                f"algorithm.adv_estimator={adv_estimator} requires "
                "actor_rollout_ref.rollout.n >= 2"
            )

        critic = config.get("critic")
        if adv_estimator in _CRITIC_ESTIMATORS:
            if not isinstance(critic, dict):
                errors.append(
                    f"algorithm.adv_estimator={adv_estimator} requires critic configuration"
                )
            elif adv_estimator.startswith("no_concat_") and critic.get("enable") is not True:
                # vendored need_critic() only infers the stock `gae` estimator;
                # custom no-concat GAE must opt in explicitly.
                errors.append(
                    f"algorithm.adv_estimator={adv_estimator} requires critic.enable=True"
                )

    # Critic configuration
    if "critic" in config:
        critic = config["critic"]

        strategy = critic.get("strategy")
        if strategy not in {"fsdp", "fsdp2", "megatron", None}:
            errors.append(f"Unknown critic strategy: {strategy}")

    # Reward configuration
    if "reward" in config:
        reward = config["reward"]

        # Check for custom reward mode
        reward_mode = reward.get("mode")
        if reward_mode and reward_mode not in {"outcome", "bounded_process", "format_gate", "custom"}:
            errors.append(f"Unknown reward mode: {reward_mode}")

    # Parity gate configuration
    if "parity_gate" in config:
        parity = config["parity_gate"]

        if parity.get("enable"):
            # Check thresholds are reasonable
            ratio_p95 = parity.get("abort_ratio_p95")
            if ratio_p95 is not None and ratio_p95 < 1.0:
                errors.append(f"Suspicious abort_ratio_p95: {ratio_p95} (usually > 1.0)")

    # Logprob parity configuration (P1: enhanced validation)
    if "logprob_parity" in config:
        parity = config["logprob_parity"]

        if parity.get("enable"):
            # Validate thresholds
            ratio_mean_threshold = parity.get("ratio_mean_threshold", 1.05)
            if not (1.0 <= ratio_mean_threshold <= 2.0):
                errors.append(f"Suspicious ratio_mean_threshold: {ratio_mean_threshold} (typically 1.01-1.1)")

            ratio_p95_threshold = parity.get("ratio_p95_threshold", 1.2)
            if not (1.0 <= ratio_p95_threshold <= 3.0):
                errors.append(f"Suspicious ratio_p95_threshold: {ratio_p95_threshold} (typically 1.1-1.5)")

    return errors


def validate_evaluation_config(config: dict[str, Any]) -> list[str]:
    """Validate evaluation configuration.

    Args:
        config: Evaluation configuration dictionary

    Returns:
        List of validation error messages (empty if valid)

    Example:
        >>> config = {"n_envs": 128, "environment": "sokoban"}
        >>> errors = validate_evaluation_config(config)
    """
    errors = []

    # Number of environments
    n_envs = config.get("n_envs")
    if n_envs is None:
        errors.append("Missing required key: n_envs")
    elif n_envs <= 0:
        errors.append(f"Invalid n_envs: {n_envs} (must be > 0)")

    # Environment name
    environment = config.get("environment")
    if not environment:
        errors.append("Missing required key: environment")

    # Model path
    model_path = config.get("model_path")
    if not model_path:
        errors.append("Missing required key: model_path")

    # Backend configuration
    backend = config.get("backend", "sglang")
    if backend not in {"sglang", "vllm", "openai", "hf"}:
        errors.append(f"Unknown backend: {backend}")

    # Observation ablation
    observation_ablation = config.get("observation_ablation", "none")
    valid_ablations = {"none", "remove", "shuffle_tiles"}
    if observation_ablation not in valid_ablations:
        errors.append(f"Unknown observation_ablation: {observation_ablation} (valid: {valid_ablations})")

    return errors


def validate_experiment_matrix_config(config: dict[str, Any]) -> list[str]:
    """Validate experiment matrix configuration.

    Args:
        config: Experiment matrix configuration (from matrix.yaml)

    Returns:
        List of validation error messages

    Example:
        >>> config = {"methods": {"concat_grpo": {"advantage": "grpo"}}, "environments": {"sokoban": {}}}
        >>> errors = validate_experiment_matrix_config(config)
    """
    errors = []

    # Methods
    methods = config.get("methods")
    if not isinstance(methods, dict) or not methods:
        errors.append("No methods specified in experiment matrix")
        methods = {}

    valid_methods = {
        "base",
        "concat_grpo",
        "no_concat_gae",
        "no_concat_episode_grpo",
        "base_eval",
    }
    for method, method_config in methods.items():
        if method not in valid_methods:
            errors.append(f"Unknown method in matrix: {method}")
            continue
        if not isinstance(method_config, dict):
            errors.append(f"Method {method} configuration must be a mapping")
            continue
        if method == "base_eval":
            continue
        if method == "base":
            if method_config.get("train") is not False or method_config.get("eval_only") is not True:
                errors.append("Method base must set train=false and eval_only=true")
            continue

        required_method_keys = {
            "advantage",
            "concat_multi_turn",
            "critic",
            "rollout_n",
        }
        missing_method_keys = sorted(required_method_keys - method_config.keys())
        if missing_method_keys:
            errors.append(
                f"Method {method} is missing keys: {', '.join(missing_method_keys)}"
            )
            continue
        if method_config["advantage"] not in _VALID_ADVANTAGE_ESTIMATORS:
            errors.append(
                f"Method {method} has unknown advantage: {method_config['advantage']}"
            )
        if not isinstance(method_config["concat_multi_turn"], bool):
            errors.append(f"Method {method} concat_multi_turn must be boolean")
        if not isinstance(method_config["critic"], bool):
            errors.append(f"Method {method} critic must be boolean")
        rollout_n = method_config["rollout_n"]
        if not isinstance(rollout_n, int) or isinstance(rollout_n, bool) or rollout_n <= 0:
            errors.append(f"Method {method} rollout_n must be a positive integer")

    # Environments
    environments = config.get("environments")
    if not isinstance(environments, dict) or not environments:
        errors.append("No environments specified in experiment matrix")
        environments = {}

    valid_envs = {"sokoban", "navigation", "frozenlake"}
    for env, env_config in environments.items():
        if env not in valid_envs:
            errors.append(f"Unknown environment: {env}")
            continue
        if not isinstance(env_config, dict):
            errors.append(f"Environment {env} configuration must be a mapping")
            continue
        required_env_keys = {
            "train",
            "train_seeds",
            "validation",
            "validation_seeds",
            "evaluation",
            "evaluation_seeds",
        }
        missing_env_keys = sorted(required_env_keys - env_config.keys())
        if missing_env_keys:
            errors.append(
                f"Environment {env} is missing keys: {', '.join(missing_env_keys)}"
            )
        for split in ("train", "validation", "evaluation"):
            path = env_config.get(split)
            if path is not None and (not isinstance(path, str) or not path.strip()):
                errors.append(f"Environment {env} {split} must be a non-empty path")
            seed_range = env_config.get(f"{split}_seeds")
            if seed_range is None:
                continue
            if (
                not isinstance(seed_range, list)
                or len(seed_range) not in {2, 3}
                or any(not isinstance(value, int) or isinstance(value, bool) for value in seed_range)
            ):
                errors.append(
                    f"Environment {env} {split}_seeds must be [start, stop] or [start, stop, step] integers"
                )
                continue
            start, stop = seed_range[:2]
            step = seed_range[2] if len(seed_range) == 3 else 1
            if start > stop or step <= 0:
                errors.append(
                    f"Environment {env} {split}_seeds has invalid range: {seed_range}"
                )

    funnel = config.get("funnel")
    if not isinstance(funnel, dict):
        errors.append("Experiment matrix funnel must be a mapping")
    else:
        for stage_name, stage in funnel.items():
            if not isinstance(stage, dict):
                errors.append(f"Funnel stage {stage_name} must be a mapping")
                continue
            for method in stage.get("methods", []):
                if method not in methods:
                    errors.append(
                        f"Funnel stage {stage_name} references unknown method: {method}"
                    )
            for env in stage.get("environments", []):
                if env not in environments:
                    errors.append(
                        f"Funnel stage {stage_name} references unknown environment: {env}"
                    )
            seeds = stage.get("seeds")
            if seeds is not None and (
                not isinstance(seeds, list)
                or not seeds
                or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
            ):
                errors.append(
                    f"Funnel stage {stage_name} seeds must be a non-empty integer list"
                )

    return errors


def check_gpu_availability(required_gpus: int) -> tuple[bool, str]:
    """Check if required GPUs are available.

    Args:
        required_gpus: Number of GPUs required

    Returns:
        Tuple of (available, message)

    Example:
        >>> available, msg = check_gpu_availability(4)
        >>> if not available:
        ...     print(f"WARNING: {msg}")
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA is not available"

        available_gpus = torch.cuda.device_count()
        if available_gpus < required_gpus:
            return False, f"Required {required_gpus} GPUs but only {available_gpus} available"

        return True, f"Found {available_gpus} GPUs"

    except ImportError:
        return False, "PyTorch not available for GPU check"


def load_validation_n_envs(config_path: str | Path) -> int:
    """Return the exact validation episode count from an environment YAML.

    The manifest is part of the experiment evidence, so malformed or ambiguous
    validation configs are rejected instead of silently falling back to a
    conventional episode count.
    """
    import yaml

    path = Path(config_path)
    try:
        config = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not parse validation config {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Validation config {path} must contain a YAML mapping")
    envs = config.get("envs")
    if not isinstance(envs, list) or not envs:
        raise ValueError(f"Validation config {path} must contain a non-empty envs list")

    total = 0
    for index, env in enumerate(envs):
        if not isinstance(env, dict):
            raise ValueError(f"Validation config {path} envs[{index}] must be a mapping")
        n_envs = env.get("n_envs")
        if not isinstance(n_envs, int) or isinstance(n_envs, bool) or n_envs <= 0:
            raise ValueError(
                f"Validation config {path} envs[{index}].n_envs must be a positive integer"
            )
        total += n_envs
    return total


def validate_model_path(model_path: str | None) -> list[str]:
    """Validate model path exists and contains required files.

    Args:
        model_path: Path to model directory

    Returns:
        List of validation errors

    Example:
        >>> errors = validate_model_path("/path/to/Qwen2.5-VL-3B")
    """
    errors = []

    if not model_path:
        errors.append("Model path is None or empty")
        return errors

    from pathlib import Path

    path = Path(model_path)

    if not path.exists():
        errors.append(f"Model path does not exist: {model_path}")
        return errors

    if not path.is_dir():
        errors.append(f"Model path is not a directory: {model_path}")
        return errors

    # Check for required files
    required_files = ["config.json"]
    for file in required_files:
        if not (path / file).exists():
            errors.append(f"Missing required file: {file}")

    # Check for model weights
    has_weights = any([
        (path / "pytorch_model.bin").exists(),
        (path / "model.safetensors").exists(),
        any(path.glob("pytorch_model-*.bin")),
        any(path.glob("model-*.safetensors")),
    ])

    if not has_weights:
        errors.append("No model weights found (looking for .bin or .safetensors)")

    return errors


def print_validation_report(
    config_type: str,
    errors: list[str],
    warnings: list[str] | None = None,
) -> bool:
    """Print validation report and return whether config is valid.

    Args:
        config_type: Type of config being validated
        errors: List of error messages
        warnings: Optional list of warning messages

    Returns:
        True if valid (no errors), False otherwise

    Example:
        >>> errors = validate_training_config(config)
        >>> valid = print_validation_report("Training", errors)
        >>> if not valid:
        ...     sys.exit(1)
    """
    print(f"\n{'='*60}")
    print(f"Configuration Validation: {config_type}")
    print(f"{'='*60}")

    if not errors and not warnings:
        print("✓ Configuration is valid")
        print(f"{'='*60}\n")
        return True

    if errors:
        print(f"\n❌ Found {len(errors)} error(s):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warning(s):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")

    print(f"\n{'='*60}\n")
    return len(errors) == 0


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Config file to validate")
    parser.add_argument("--type", choices=["training", "evaluation", "matrix"], required=True)
    args = parser.parse_args()

    # Load config
    import yaml

    with args.config.open() as f:
        config = yaml.safe_load(f)

    # Validate based on type
    if args.type == "training":
        errors = validate_training_config(config)
    elif args.type == "evaluation":
        errors = validate_evaluation_config(config)
    elif args.type == "matrix":
        errors = validate_experiment_matrix_config(config)
    else:
        raise ValueError(f"Unknown config type: {args.type}")

    # Print report
    valid = print_validation_report(args.type.capitalize(), errors)

    sys.exit(0 if valid else 1)
