import pytest

from vagen.utils.multimodal_support import (
    is_qwen2_vl_image_processor,
    require_supported_no_concat_processor,
)


Qwen2Processor = type("Qwen2VLImageProcessorFast", (), {})
Qwen3Processor = type("Qwen3VLImageProcessor", (), {})


def test_qwen25_vl_uses_verified_no_concat_mrope_path():
    processor = Qwen2Processor()
    assert is_qwen2_vl_image_processor(processor)
    require_supported_no_concat_processor(processor)


def test_qwen3_vl_fails_loudly_instead_of_silent_1d_position_ids():
    with pytest.raises(NotImplementedError, match="Qwen3-VL"):
        require_supported_no_concat_processor(Qwen3Processor())
