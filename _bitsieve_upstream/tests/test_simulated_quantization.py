"""The bit-width sweep needs a quantizer the packed kernels cannot provide.

The packed path addresses 2 and 4 bits only - 3-bit values straddle byte
boundaries. Their quantization GRID is well defined though, so a coverage study
(which cares what the selector sees, not how it is stored) can use a
round-tripped quantizer instead. That is only trustworthy if it is bit-exact
with the packed path wherever the packed path exists, which is what these pin.
"""
import pytest
import torch

from bitsieve_fastdllm.reference import (
    dequantize_keys,
    quantize_keys,
    simulate_key_quantization,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("bits", [2, 4])
def test_bit_exact_with_the_packed_path(dtype, bits):
    torch.manual_seed(0)
    key = torch.randn(2, 4, 64, 128, dtype=dtype)
    packed = dequantize_keys(
        quantize_keys(key, bits=bits, token_group=32), dtype=torch.float32
    )
    simulated = simulate_key_quantization(
        key, bits=bits, token_group=32, dtype=torch.float32
    )
    # Exactly equal, not merely close: the dtype the min/max are taken in and the
    # float16 storage of scale/zero each move values a whole quantization step
    # near a bin edge, so "close" would mean the simulation is a different
    # quantizer that happens to agree on average.
    assert torch.equal(packed, simulated)


def test_three_bits_works_where_the_packed_path_refuses():
    key = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        quantize_keys(key, bits=3, token_group=32)
    out = simulate_key_quantization(key, bits=3, token_group=32)
    assert out.shape == key.shape
    assert out.dtype == key.dtype


def test_error_falls_monotonically_with_bit_width():
    torch.manual_seed(0)
    key = torch.randn(1, 4, 128, 128, dtype=torch.float32)
    errs = [
        (simulate_key_quantization(key, bits=b, token_group=32) - key).abs().mean().item()
        for b in (2, 3, 4, 5, 8)
    ]
    assert errs == sorted(errs, reverse=True), errs


def test_sixteen_bits_is_the_identity():
    key = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16)
    assert torch.equal(simulate_key_quantization(key, bits=16, token_group=32), key)


def test_rejects_shapes_and_widths_it_cannot_honour():
    key = torch.randn(1, 2, 33, 64)
    with pytest.raises(ValueError):
        simulate_key_quantization(key, bits=4, token_group=32)   # T not divisible
    with pytest.raises(ValueError):
        simulate_key_quantization(torch.randn(1, 2, 32), bits=4)  # wrong rank
    with pytest.raises(ValueError):
        simulate_key_quantization(torch.randn(1, 2, 32, 64), bits=17)


def test_ragged_tail_is_refused_by_default_and_handled_on_request():
    key = torch.randn(1, 2, 70, 64, dtype=torch.float32)   # 70 = 2*32 + 6
    with pytest.raises(ValueError):
        simulate_key_quantization(key, bits=4, token_group=32)
    out = simulate_key_quantization(key, bits=4, token_group=32, allow_ragged=True)
    assert out.shape == key.shape
    # The whole groups must be untouched by the presence of a tail.
    head = simulate_key_quantization(key[:, :, :64, :], bits=4, token_group=32)
    assert torch.equal(out[:, :, :64, :], head)
    # And the tail must actually be quantized, not passed through.
    assert not torch.equal(out[:, :, 64:, :], key[:, :, 64:, :])
