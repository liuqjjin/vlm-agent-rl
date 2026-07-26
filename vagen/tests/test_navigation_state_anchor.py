from __future__ import annotations

import json

from vagen.envs.navigation.navigation_env import navigation_state_anchor
from vagen.envs_remote.gym_image_env_client import _decode_observation
from vagen.envs_remote.handler import BaseGymHandler
from vagen.utils.state_anchor import canonical_state_anchor


def _agent(yaw: float = 90.0) -> dict:
    return {
        "position": {"x": 1.20000001, "y": 0.90000001, "z": -0.29999999},
        "rotation": {"x": 0.0, "y": yaw, "z": 0.0},
        "cameraHorizon": 30.00000001,
        "isStanding": True,
    }


def test_navigation_anchor_is_stable_text_and_preserves_orientation():
    anchor = navigation_state_anchor(_agent())
    decoded = json.loads(anchor)
    assert decoded == {
        "camera_horizon": 30.0,
        "position": {"x": 1.2, "y": 0.9, "z": -0.3},
        "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
        "standing": True,
    }
    assert navigation_state_anchor(_agent(180.0)) != anchor
    assert canonical_state_anchor({"state_anchor": anchor}, 4).endswith(
        "[remaining_turns=4]"
    )


def test_remote_protocol_round_trips_state_anchor():
    wire = BaseGymHandler._obs_to_result(
        {"obs_str": "<image>", "state_anchor": "pose=1"}
    )
    assert wire.data == {"obs": "<image>", "state_anchor": "pose=1"}
    restored = _decode_observation(wire.data, images=None)
    assert restored == {"obs_str": "<image>", "state_anchor": "pose=1"}


def test_remote_protocol_does_not_invent_anchor_for_other_environments():
    wire = BaseGymHandler._obs_to_result({"obs_str": "plain"})
    assert wire.data == {"obs": "plain"}
    assert _decode_observation(wire.data, images=None) == {"obs_str": "plain"}
