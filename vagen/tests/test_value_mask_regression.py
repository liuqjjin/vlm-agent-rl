"""Regression tests for sparse no-concat critic supervision.

The no-concat GAE estimators supervise one value prediction per turn and mark
all other return positions with ``-100``.  Both the trainer and the critic
batch-selection path must preserve the resulting ``value_mask``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_value_loss
from vagen.custom_advantage.no_concat_gae import (
    compute_gae_no_concat_advantage_return_firsttok,
)


ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = ROOT / "vagen" / "ray_trainer.py"


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _value_mask_estimators() -> set[str]:
    """Read the dispatch guard that immediately attaches ``compute_value_mask``."""
    fit = _function_node(TRAINER_PATH, "fit")
    for node in ast.walk(fit):
        if not isinstance(node, ast.If):
            continue
        calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)]
        if not any(isinstance(call.func, ast.Name) and call.func.id == "compute_value_mask" for call in calls):
            continue
        return {child.value for child in ast.walk(node.test) if isinstance(child, ast.Constant) and isinstance(child.value, str)}
    raise AssertionError("could not find the value-mask dispatch in RayPPOTrainer.fit")


def _sparse_critic_batch() -> DataProto:
    returns = torch.tensor([[-100.0, 2.0]])
    batch = TensorDict(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "responses": torch.tensor([[2, 3]]),
            "response_mask": torch.ones(1, 2),
            "attention_mask": torch.ones(1, 3),
            "position_ids": torch.arange(3).unsqueeze(0),
            "values": torch.zeros(1, 2),
            "returns": returns,
            "value_mask": returns.ne(-100.0).float(),
        },
        batch_size=[1],
    )
    return DataProto(batch=batch, non_tensor_batch={}, meta_info={})


def test_current_no_concat_gae_name_enables_value_mask():
    assert "no_concat_gae" in _value_mask_estimators()


def test_value_mask_survives_critic_selection_and_blocks_sentinel_gradient():
    from verl.workers.critic.dp_critic import _critic_update_batch_keys

    data = _sparse_critic_batch()
    selected = data.select(batch_keys=_critic_update_batch_keys(data))

    response_mask = selected.batch["response_mask"]
    effective_mask = response_mask * selected.batch.get("value_mask", response_mask)
    vpreds = torch.zeros_like(selected.batch["returns"], requires_grad=True)
    loss, _ = compute_value_loss(
        vpreds=vpreds,
        values=selected.batch["values"],
        returns=selected.batch["returns"],
        response_mask=effective_mask,
        cliprange_value=10.0,
        loss_agg_mode="token-mean",
    )
    loss.backward()

    assert vpreds.grad[0, 0].item() == pytest.approx(0.0)
    assert abs(vpreds.grad[0, 1].item()) > 0
    assert effective_mask.sum().item() == 1


def test_no_concat_gae_supervises_exactly_one_value_per_turn():
    batch = TensorDict(
        {
            "token_level_scores": torch.tensor([[0.0, 0.2, 0.0], [0.0, 1.2, 0.0]]),
            "values": torch.zeros(2, 3),
            "response_mask": torch.tensor([[0, 1, 0], [0, 1, 0]]),
        },
        batch_size=[2],
    )
    data = DataProto(
        batch=batch,
        non_tensor_batch={
            "group_idx": np.array(["g", "g"], dtype=object),
            "traj_idx": np.array([0, 0]),
            "turn_idx": np.array([1, 2]),
        },
        meta_info={},
    )

    _, returns = compute_gae_no_concat_advantage_return_firsttok(data, gamma=1.0, lam=1.0)
    mask = returns.ne(-100.0)
    assert mask.sum().item() == 2
    assert mask.sum(dim=-1).tolist() == [1, 1]


def test_sparse_value_supervision_remains_masked_for_twenty_updates():
    returns = torch.tensor([[-100.0, 2.0]])
    value_mask = returns.ne(-100.0).float()
    predictions = torch.nn.Parameter(torch.tensor([[0.5, -1.0]]))
    optimizer = torch.optim.SGD([predictions], lr=0.2)
    ignored_initial = predictions.detach()[0, 0].item()

    for _ in range(20):
        optimizer.zero_grad()
        loss, _ = compute_value_loss(
            vpreds=predictions,
            values=predictions.detach(),
            returns=returns,
            response_mask=value_mask,
            cliprange_value=1000.0,
            loss_agg_mode="token-mean",
        )
        loss.backward()
        assert predictions.grad[0, 0].item() == pytest.approx(0.0)
        optimizer.step()

    assert predictions.detach()[0, 0].item() == pytest.approx(ignored_initial)
    assert predictions.detach()[0, 1].item() == pytest.approx(2.0, abs=0.04)


def test_value_mask_drives_real_critic_optimizer_update() -> None:
    from verl.workers.critic.dp_critic import DataParallelPPOCritic

    class _TwoTokenCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.predictions = torch.nn.Parameter(torch.tensor([0.5, -1.0]))

    config = OmegaConf.create(
        {
            "model": {"use_remove_padding": False},
            "ulysses_sequence_parallel_size": 1,
            "ppo_mini_batch_size": 1,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_epochs": 1,
            "use_dynamic_bsz": False,
            "cliprange_value": 1000.0,
            "loss_agg_mode": "token-mean",
            "grad_clip": 100.0,
        }
    )
    module = _TwoTokenCritic()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.2)
    critic = DataParallelPPOCritic(config, module, optimizer)

    def _forward(self, model_inputs):
        batch_size = model_inputs["responses"].shape[0]
        return self.critic_module.predictions.unsqueeze(0).expand(batch_size, -1)

    critic._forward_micro_batch = MethodType(_forward, critic)
    before = module.predictions.detach().clone()
    metrics = critic.update_critic(_sparse_critic_batch())

    assert module.predictions.detach()[0].item() == pytest.approx(before[0].item())
    assert module.predictions.detach()[1].item() != pytest.approx(before[1].item())
    assert "critic/vf_loss" in metrics
