"""Edge case tests for state anchor extraction and canonicalization."""

from __future__ import annotations

import json

import pytest


def _implementation():
    from vagen.utils.state_anchor import canonical_state_anchor
    return canonical_state_anchor


def test_canonical_state_anchor_handles_missing_anchor():
    """Missing state_anchor should return None."""
    canonical = _implementation()
    result = canonical({"observation": "data"}, remaining_turns=5)
    assert result is None


def test_canonical_state_anchor_handles_none_anchor():
    """Explicit None state_anchor should return None."""
    canonical = _implementation()
    result = canonical({"state_anchor": None}, remaining_turns=5)
    assert result is None


def test_canonical_state_anchor_preserves_string_anchors():
    """String anchors should be preserved and normalized."""
    canonical = _implementation()
    result = canonical({"state_anchor": "  pose=1  "}, remaining_turns=3)
    assert result == "pose=1\n[remaining_turns=3]"


def test_canonical_state_anchor_handles_whitespace_only_strings():
    """Whitespace-only strings should be treated as empty after stripping."""
    canonical = _implementation()
    result = canonical({"state_anchor": "   "}, remaining_turns=2)
    assert result == "\n[remaining_turns=2]"


def test_canonical_state_anchor_serializes_dict_anchors():
    """Dict anchors should be serialized to stable JSON."""
    canonical = _implementation()
    anchor = {"position": [1.2, 0.9], "rotation": 90}
    result = canonical({"state_anchor": anchor}, remaining_turns=4)

    # Should be valid JSON with sorted keys
    lines = result.split("\n")
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed == anchor
    assert lines[1] == "[remaining_turns=4]"


def test_canonical_state_anchor_serializes_nested_structures():
    """Nested dicts and lists should be serialized correctly."""
    canonical = _implementation()
    anchor = {
        "agent": {"x": 1.0, "y": 2.0},
        "objects": [{"id": 1}, {"id": 2}],
    }
    result = canonical({"state_anchor": anchor}, remaining_turns=1)

    lines = result.split("\n")
    parsed = json.loads(lines[0])
    assert parsed == anchor


def test_canonical_state_anchor_handles_unbounded_turns():
    """None remaining_turns should produce 'unbounded' budget."""
    canonical = _implementation()
    result = canonical({"state_anchor": "state"}, remaining_turns=None)
    assert result == "state\n[remaining_turns=unbounded]"


def test_canonical_state_anchor_clamps_negative_turns_to_zero():
    """Negative remaining_turns should be clamped to 0."""
    canonical = _implementation()
    result = canonical({"state_anchor": "state"}, remaining_turns=-5)
    assert result == "state\n[remaining_turns=0]"


def test_canonical_state_anchor_handles_zero_turns():
    """Zero remaining turns should be valid."""
    canonical = _implementation()
    result = canonical({"state_anchor": "state"}, remaining_turns=0)
    assert result == "state\n[remaining_turns=0]"


def test_canonical_state_anchor_handles_large_turn_counts():
    """Large turn counts should be handled without overflow."""
    canonical = _implementation()
    result = canonical({"state_anchor": "state"}, remaining_turns=1000000)
    assert result == "state\n[remaining_turns=1000000]"


def test_canonical_state_anchor_preserves_unicode():
    """Unicode characters should be preserved in output."""
    canonical = _implementation()
    anchor = {"location": "位置A", "emoji": "🎯"}
    result = canonical({"state_anchor": anchor}, remaining_turns=2)

    lines = result.split("\n")
    parsed = json.loads(lines[0])
    assert parsed == anchor


def test_canonical_state_anchor_handles_numeric_anchors():
    """Numeric anchors should be serialized to JSON."""
    canonical = _implementation()
    result = canonical({"state_anchor": 42}, remaining_turns=3)
    assert result == "42\n[remaining_turns=3]"


def test_canonical_state_anchor_handles_boolean_anchors():
    """Boolean anchors should be serialized to JSON."""
    canonical = _implementation()
    result = canonical({"state_anchor": True}, remaining_turns=1)
    assert result == "true\n[remaining_turns=1]"


def test_canonical_state_anchor_handles_list_anchors():
    """List anchors should be serialized to JSON."""
    canonical = _implementation()
    anchor = [1, 2, 3]
    result = canonical({"state_anchor": anchor}, remaining_turns=2)
    assert result == "[1,2,3]\n[remaining_turns=2]"


def test_canonical_state_anchor_produces_stable_json_ordering():
    """JSON serialization should use sorted keys for stability."""
    canonical = _implementation()
    # Keys intentionally out of order
    anchor1 = {"z": 1, "a": 2, "m": 3}
    anchor2 = {"m": 3, "z": 1, "a": 2}

    result1 = canonical({"state_anchor": anchor1}, remaining_turns=1)
    result2 = canonical({"state_anchor": anchor2}, remaining_turns=1)

    # Should produce identical output
    assert result1 == result2


def test_canonical_state_anchor_handles_float_precision():
    """Floating point numbers should be serialized consistently."""
    canonical = _implementation()
    anchor = {"value": 1.23456789}
    result = canonical({"state_anchor": anchor}, remaining_turns=1)

    lines = result.split("\n")
    parsed = json.loads(lines[0])
    # JSON should preserve precision
    assert parsed["value"] == 1.23456789


def test_canonical_state_anchor_rejects_non_serializable_objects():
    """Non-JSON-serializable objects should raise an error."""
    canonical = _implementation()

    class CustomObject:
        pass

    with pytest.raises((TypeError, ValueError)):
        canonical({"state_anchor": CustomObject()}, remaining_turns=1)


def test_canonical_state_anchor_different_turns_produce_different_keys():
    """Same state with different remaining turns should produce different keys."""
    canonical = _implementation()
    result1 = canonical({"state_anchor": "state"}, remaining_turns=1)
    result2 = canonical({"state_anchor": "state"}, remaining_turns=2)

    assert result1 != result2
    assert result1 == "state\n[remaining_turns=1]"
    assert result2 == "state\n[remaining_turns=2]"


def test_canonical_state_anchor_handles_float_remaining_turns():
    """Float remaining_turns should be converted to int."""
    canonical = _implementation()
    result = canonical({"state_anchor": "state"}, remaining_turns=3.7)
    assert result == "state\n[remaining_turns=3]"


def test_canonical_state_anchor_empty_dict_anchor():
    """Empty dict anchor should be serialized correctly."""
    canonical = _implementation()
    result = canonical({"state_anchor": {}}, remaining_turns=1)
    assert result == "{}\n[remaining_turns=1]"


def test_canonical_state_anchor_empty_string_anchor():
    """Empty string anchor should be valid."""
    canonical = _implementation()
    result = canonical({"state_anchor": ""}, remaining_turns=1)
    assert result == "\n[remaining_turns=1]"


def test_navigation_state_anchor_integration():
    """Integration test with actual navigation environment anchor format."""
    from vagen.envs.navigation.navigation_env import navigation_state_anchor

    agent_state = {
        "position": {"x": 1.20000001, "y": 0.90000001, "z": -0.29999999},
        "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
        "cameraHorizon": 30.00000001,
        "isStanding": True,
    }

    anchor = navigation_state_anchor(agent_state)
    canonical = _implementation()
    result = canonical({"state_anchor": anchor}, remaining_turns=5)

    # Should produce a valid canonical key
    assert "[remaining_turns=5]" in result

    # Should be parseable JSON
    lines = result.split("\n")
    parsed = json.loads(lines[0])
    assert "position" in parsed
    assert parsed["position"]["x"] == 1.2  # Rounded
