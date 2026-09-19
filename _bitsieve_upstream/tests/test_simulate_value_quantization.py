"""simulate_value_quantization must be bit-exact with the packed value path.

Training degrades old-cache values through the simulate_* function; if it drifts
from quantize_values + dequantize_values, the student would be learning to
tolerate a grid the cache never produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.reference import (  # noqa: E402
    dequantize_values,
    quantize_values,
    simulate_value_quantization,
)


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("channel_group", [32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matches_packed_path(bits: int, channel_group: int, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    value = torch.randn(2, 4, 96, 128, dtype=dtype)

    packed = quantize_values(value, bits=bits, channel_group=channel_group)
    via_packed = dequantize_values(packed, dtype=dtype)
    simulated = simulate_value_quantization(
        value, bits=bits, channel_group=channel_group, dtype=dtype
    )
    assert torch.equal(simulated, via_packed)


def test_16_bits_is_identity() -> None:
    value = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16)
    assert torch.equal(simulate_value_quantization(value, bits=16), value)


def test_error_shrinks_with_more_bits() -> None:
    torch.manual_seed(0)
    value = torch.randn(1, 4, 64, 128, dtype=torch.float32)
    err = [
        (simulate_value_quantization(value, bits=b) - value).abs().mean().item()
        for b in (2, 3, 4, 8)
    ]
    assert err == sorted(err, reverse=True), err


def test_rejects_indivisible_head_dim() -> None:
    value = torch.randn(1, 1, 8, 40, dtype=torch.float32)
    with pytest.raises(ValueError, match="channel_group"):
        simulate_value_quantization(value, bits=4, channel_group=32)
