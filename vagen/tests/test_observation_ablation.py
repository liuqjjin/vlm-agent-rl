from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vagen.evaluate.observation_ablation import ablate_images, shuffle_image_tiles
from vagen.evaluate.run_eval import _job_resume_key, _load_config, _parse_env_specs
from vagen.evaluate.vision_workflow import GenericVisionInferenceWorkflow


def _image() -> Image.Image:
    values = np.arange(100, dtype=np.uint8).reshape(10, 10)
    return Image.fromarray(values)


def test_remove_ablation_drops_all_images_without_mutating_input():
    image = _image()
    assert ablate_images([image], mode="remove", seed=3, turn=0) == []
    assert np.asarray(image)[0, 0] == 0


def test_tile_shuffle_is_deterministic_and_preserves_pixel_multiset():
    image = _image()
    first = shuffle_image_tiles(
        image,
        seed=9,
        turn=2,
        image_index=0,
        grid_size=5,
    )
    second = shuffle_image_tiles(
        image,
        seed=9,
        turn=2,
        image_index=0,
        grid_size=5,
    )
    assert first.size == image.size
    assert np.array_equal(first, second)
    assert not np.array_equal(first, image)
    assert sorted(np.asarray(first).reshape(-1).tolist()) == sorted(
        np.asarray(image).reshape(-1).tolist()
    )


def test_unknown_ablation_fails_closed():
    with pytest.raises(ValueError, match="unknown observation ablation"):
        ablate_images([_image()], mode="blur", seed=0, turn=0)


def test_eval_config_records_ablation_and_resume_keys_do_not_collide():
    specs = _parse_env_specs(
        {
            "envs": [
                {
                    "name": "Sokoban",
                    "n_envs": 1,
                    "tag_id": "anti_cheat",
                    "observation_ablation": "shuffle_tiles",
                }
            ]
        }
    )
    assert specs[0].observation_ablation == "shuffle_tiles"
    base = {"env_name": "Sokoban", "seed": 7, "tag_id": "anti_cheat"}
    assert _job_resume_key({**base, "observation_ablation": "none"}) != (
        _job_resume_key({**base, "observation_ablation": "remove"})
    )

    with pytest.raises(ValueError, match="unknown observation_ablation"):
        _parse_env_specs(
            {
                "envs": [
                    {
                        "name": "Sokoban",
                        "n_envs": 1,
                        "tag_id": "bad",
                        "observation_ablation": "blur",
                    }
                ]
            }
        )


@pytest.mark.asyncio
async def test_workflow_applies_removal_before_model_and_records_it(tmp_path):
    class Adapter:
        def __init__(self):
            self.user_image_counts = []

        def format_system(self, text, images):
            return {"role": "system", "content": text}

        def format_user_turn(self, text, images):
            self.user_image_counts.append(len(images))
            return {"role": "user", "content": text}

        def format_assistant_turn(self, text):
            return {"role": "assistant", "content": text}

        async def acompletion(self, messages, **kwargs):
            return "<answer>right</answer>"

    class Env:
        def __init__(self, config):
            pass

        async def reset(self, seed):
            return {
                "obs_str": "<image>",
                "multi_modal_input": {"<image>": [_image()]},
            }, {}

        async def system_prompt(self):
            return {"obs_str": "act"}

        async def step(self, action):
            return {
                "obs_str": "<image>",
                "multi_modal_input": {"<image>": [_image()]},
            }, 1.0, True, {"success": True}

        async def close(self):
            pass

    adapter = Adapter()
    workflow = GenericVisionInferenceWorkflow(
        adapter,
        dump_dir=str(tmp_path),
        observation_ablation="remove",
    )
    result = await workflow.arun_episode(
        Env,
        {},
        seed=4,
        max_turns=1,
    )
    assert result["success"]
    assert adapter.user_image_counts == [0, 0]
    metrics_paths = list(tmp_path.glob("*/metrics.json"))
    assert len(metrics_paths) == 1
    assert '"observation_ablation": "remove"' in metrics_paths[0].read_text()


def test_indexed_eval_override_updates_env_list_in_place(tmp_path):
    config = tmp_path / "eval.yaml"
    config.write_text(
        "envs:\n"
        "  - name: Sokoban\n"
        "    n_envs: 60\n"
        "    tag_id: base\n"
        "    observation_ablation: none\n"
    )
    loaded = _load_config(
        str(config),
        [
            "envs.0.n_envs=2",
            "envs.0.observation_ablation=shuffle_tiles",
        ],
    )
    assert loaded.envs[0].n_envs == 2
    assert loaded.envs[0].observation_ablation == "shuffle_tiles"
