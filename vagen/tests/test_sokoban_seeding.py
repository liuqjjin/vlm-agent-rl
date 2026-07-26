from __future__ import annotations

import os
import subprocess
import sys

from vagen.envs.sokoban.patch_sokoban_env import next_sokoban_seed


def _retry_sequence(seed: int, length: int = 5) -> list[int]:
    values = []
    for _ in range(length):
        seed = next_sokoban_seed(seed)
        values.append(seed)
    return values


def test_retry_seed_sequence_is_repeatable():
    assert _retry_sequence(19) == _retry_sequence(19)
    assert len(set(_retry_sequence(19))) == 5


def test_retry_seed_sequence_is_independent_of_python_hash_salt():
    code = (
        "from vagen.envs.sokoban.patch_sokoban_env import next_sokoban_seed as n;"
        "s=19;"
        "print(','.join(str(s := n(s)) for _ in range(5)))"
    )
    outputs = []
    for salt in ("1", "99991"):
        env = {**os.environ, "PYTHONHASHSEED": salt}
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]
