"""Stable metadata keys for exact state-relative grouping."""

from __future__ import annotations

import json
from typing import Any


def canonical_state_anchor(obs: dict[str, Any], remaining_turns: int | None) -> str | None:
    """Build a text-only pre-action state key with the remaining turn budget."""
    raw_anchor = obs.get("state_anchor")
    if raw_anchor is None:
        return None
    if isinstance(raw_anchor, str):
        anchor = raw_anchor.strip()
    else:
        anchor = json.dumps(raw_anchor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    budget = "unbounded" if remaining_turns is None else str(max(0, int(remaining_turns)))
    return f"{anchor}\n[remaining_turns={budget}]"

