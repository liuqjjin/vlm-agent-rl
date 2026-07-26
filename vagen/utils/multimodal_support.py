"""Explicit support boundaries for no-concat multimodal training."""

from __future__ import annotations


def is_qwen2_vl_image_processor(image_processor: object) -> bool:
    """Return whether the processor uses the verified Qwen2/2.5-VL M-RoPE path."""
    return "Qwen2VLImageProcessor" in image_processor.__class__.__name__


def require_supported_no_concat_processor(image_processor: object) -> None:
    if not is_qwen2_vl_image_processor(image_processor):
        name = image_processor.__class__.__name__
        raise NotImplementedError(
            "no-concat visual training currently verifies only Qwen2/2.5-VL processors; "
            f"got {name}. Qwen3-VL needs a dedicated processor/M-RoPE parity adaptation first."
        )
