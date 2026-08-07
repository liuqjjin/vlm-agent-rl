"""Production-path regression tests for no-concat validation padding."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from vagen.agent_loop.agent_loop_no_concat import (
    AgentLoopMetrics,
    AgentLoopWorkerBase,
    _InternalAgentLoopOutput,
)
from vagen.ray_trainer import (
    RayPPOTrainer,
    filter_no_concat_validation_padding,
    pad_no_concat_validation_batch,
)
from vagen.utils.concat_val_multi_turn import concat_val_multi_turn


class _Tokenizer:
    pad_token_id = 0


def _input_batch(size: int) -> DataProto:
    return DataProto(
        batch=TensorDict(
            {"row": torch.arange(size, dtype=torch.long).unsqueeze(-1)},
            batch_size=[size],
        ),
        non_tensor_batch={
            "uid": np.array([f"uid-{index}" for index in range(size)], dtype=object),
        },
        meta_info={},
    )


def _turn_output(
    *,
    uid: str,
    turn_idx: int,
    num_turns: int,
    is_padding: bool,
) -> _InternalAgentLoopOutput:
    prompt_ids = torch.tensor([[0, 11]], dtype=torch.long)
    response_ids = torch.tensor([[100 + turn_idx, 0]], dtype=torch.long)
    input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    response_mask = torch.tensor([[1, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.long)
    return _InternalAgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        position_ids=torch.arange(input_ids.shape[-1]).unsqueeze(0),
        response_mask=response_mask,
        attention_mask=attention_mask,
        response_logprobs=None,
        multi_modal_inputs=None,
        multi_modal_data={},
        reward_score=float(turn_idx == num_turns),
        num_turns=1,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "reward_extra_info": {"traj_success": float(turn_idx == num_turns)},
            "image_data": [],
            "last_turn": turn_idx == num_turns,
            "group_idx": uid,
            "traj_idx": 0,
            "turn_idx": turn_idx,
            "state_anchor": f"{uid}:turn-{turn_idx}",
            "_is_padding": is_padding,
        },
    )


def _generate_variable_turn_outputs(padded_inputs: DataProto) -> DataProto:
    outputs: list[_InternalAgentLoopOutput] = []
    for row, uid in enumerate(padded_inputs.non_tensor_batch["uid"]):
        original_index = int(str(uid).split("-")[-1])
        num_turns = original_index % 3 + 1
        is_padding = bool(padded_inputs.non_tensor_batch["_is_padding"][row])
        outputs.extend(
            _turn_output(
                uid=str(uid),
                turn_idx=turn_idx,
                num_turns=num_turns,
                is_padding=is_padding,
            )
            for turn_idx in range(1, num_turns + 1)
        )

    worker = object.__new__(AgentLoopWorkerBase)
    return worker._postprocess(outputs)


def test_real_dataproto_padding_survives_variable_turn_generation() -> None:
    original = _input_batch(30)
    padded, pad_size = pad_no_concat_validation_batch(original, size_divisor=8)

    assert pad_size == 2
    assert len(padded) == 32
    # The vendored production helper copies the first rows, then appends them.
    assert padded.non_tensor_batch["uid"][-2:].tolist() == ["uid-0", "uid-1"]
    assert padded.non_tensor_batch["_is_padding"].tolist() == [False] * 30 + [True, True]

    generated_turns = _generate_variable_turn_outputs(padded)
    assert "_is_padding" in generated_turns.non_tensor_batch
    assert generated_turns.non_tensor_batch["_is_padding"].dtype == object

    filtered_turns = filter_no_concat_validation_padding(generated_turns)
    expected_turns = sum(index % 3 + 1 for index in range(30))
    assert len(filtered_turns) == expected_turns
    assert not any(filtered_turns.non_tensor_batch["_is_padding"])

    trajectories = concat_val_multi_turn(filtered_turns, original, _Tokenizer())
    assert len(trajectories) == 30
    assert trajectories.non_tensor_batch["group_idx"].tolist() == original.non_tensor_batch["uid"].tolist()
    assert trajectories.non_tensor_batch["__num_turns__"].tolist() == [
        index % 3 + 1 for index in range(30)
    ]


def test_real_dataproto_padding_handles_zero_pad_size() -> None:
    original = _input_batch(32)
    padded, pad_size = pad_no_concat_validation_batch(original, size_divisor=8)

    assert padded is original
    assert pad_size == 0
    assert padded.non_tensor_batch["_is_padding"].tolist() == [False] * 32


def test_padding_filter_fails_closed_when_sentinel_is_missing() -> None:
    with pytest.raises(ValueError, match="requires '_is_padding'"):
        filter_no_concat_validation_padding(_input_batch(2))


def test_padding_filter_rejects_non_boolean_sentinel() -> None:
    data = _input_batch(2)
    data.non_tensor_batch["_is_padding"] = np.array([False, None], dtype=object)
    with pytest.raises(TypeError, match="must contain booleans"):
        filter_no_concat_validation_padding(data)


def test_validation_jsonl_persists_authoritative_trajectory_turn_counts(
    tmp_path,
) -> None:
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_steps = 7
    trainer._dump_generations(
        inputs=["prompt-a", "prompt-b"],
        outputs=["answer-a", "answer-b"],
        images=[],
        gts=[None, None],
        scores=[1.0, 0.0],
        reward_extra_infos_dict={"num_turns": [2, 5]},
        dump_path=str(tmp_path),
    )

    records = [json.loads(line) for line in (tmp_path / "7.jsonl").read_text().splitlines()]
    assert [record["num_turns"] for record in records] == [2, 5]
