from __future__ import annotations

import pytest

from vagen.agent_loop.gym_agent_loop_no_concat import (
    AgentData,
    AgentState,
    GymAgentLoop,
)


class _SuccessWithoutDoneEnv:
    async def step(self, action: str):
        assert action == "move"
        return {"obs_str": "next"}, 1.1, False, {"success": True}


@pytest.mark.asyncio
async def test_no_concat_loop_terminates_on_success_without_done() -> None:
    loop = object.__new__(GymAgentLoop)
    loop.env_max_turns = 5
    loop.prompt_length = 8
    loop.response_length = 8
    data = AgentData(
        metrics={},
        request_id="request",
        env=_SuccessWithoutDoneEnv(),
        response_limit=8,
        env_name="fake",
        sys_images=[],
        cur_images=[],
        cur_anchor="anchor",
        group_idx=0,
        traj_idx=0,
    )
    data.last_assistant_text = "move"
    data.turn_prompt_ids = [10, 20]
    data.turn_response_mask = [1]
    data.turn_response_logprobs = [-0.1]

    state = await loop._handle_env_state(data)

    assert state is AgentState.TERMINATED
    assert len(data.outputs) == 1
    assert data.outputs[0].extra_fields["last_turn"] is True
    assert data.outputs[0].extra_fields["reward_extra_info"]["traj_success"] == 1.0
