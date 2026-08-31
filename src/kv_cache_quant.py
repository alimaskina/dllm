"""Scalar int4/int8 quantization for Fast-dLLM v2 block-level KV cache."""

from __future__ import annotations

import math

import torch
from transformers.cache_utils import DynamicCache

_QMAX = {2: 1, 4: 7, 8: 127}
MICRO_GROUP = 16
KIVI_SCHEME = "kivi"
PER_TOKEN_SCHEME = "per_token"
HADAMARD_PER_TOKEN_SCHEME = "hadamard_per_token"
QUAROT_SCHEME = "quarot"
QJL_SCHEME = "qjl"
K_QUANT_SCHEMES = (
    KIVI_SCHEME,
    PER_TOKEN_SCHEME,
    HADAMARD_PER_TOKEN_SCHEME,
    QUAROT_SCHEME,
    QJL_SCHEME,
)
V_QUANT_SCHEMES = (KIVI_SCHEME, HADAMARD_PER_TOKEN_SCHEME)
DEFAULT_KIVI_GROUP_SIZE = 32
DEFAULT_KIVI_RESIDUAL_LENGTH = 32
HADAMARD_SIGN_SEED = 0


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_keys(key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to keys [B, H, S, D]; cos/sin [B, S, D]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return key_states * cos + rotate_half(key_states) * sin


def inverse_rope_keys(key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Undo RoPE on keys (rotation by -angle)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return key_states * cos + rotate_half(key_states) * (-sin)


def rope_cos_sin(model, seq_len: int, batch_size: int, device: torch.device, dtype: torch.dtype):
    """Compute cos/sin for positions [0, seq_len) using the model's rotary embedding."""
    rotary_emb = model.model.rotary_emb
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    dummy = torch.zeros(batch_size, seq_len, model.config.hidden_size, device=device, dtype=dtype)
    return rotary_emb(dummy, position_ids)


def quantize_per_block(
    tensor: torch.Tensor,
    block_size: int,
    num_blocks: int,
    bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block symmetric scalar quantization along the sequence dimension."""
    if bits not in _QMAX:
        raise ValueError(f"bits must be one of {list(_QMAX)}, got {bits}")

    qmax = _QMAX[bits]
    b, h, _s, d = tensor.shape
    blocks = tensor[..., : num_blocks * block_size, :].view(
        b, h, num_blocks, block_size, d
    )
    amax = blocks.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    scale = amax / float(qmax)
    q = (blocks / scale).round().clamp(-qmax - 1, qmax).to(torch.int8)
    q = q.view(b, h, num_blocks * block_size, d)
    return q, scale.squeeze(-1).squeeze(-1)


def dequantize_per_block(
    q: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Dequantize per-block int4/int8 KV back to floating point."""
    b, h, s, d = q.shape
    num_blocks = s // block_size
    blocks = q.view(b, h, num_blocks, block_size, d).to(scale.dtype)
    scale = scale.unsqueeze(-1).unsqueeze(-1)
    return (blocks * scale).view(b, h, num_blocks * block_size, d)


def quantize_micro16(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric int quant with one scale per 16 elements along head_dim."""
    if bits not in _QMAX:
        raise ValueError(f"bits must be one of {list(_QMAX)}, got {bits}")
    qmax = _QMAX[bits]
    b, h, s, d = tensor.shape
    if d % MICRO_GROUP != 0:
        raise ValueError(f"head_dim {d} must be divisible by {MICRO_GROUP}")
    g = d // MICRO_GROUP
    x = tensor.view(b, h, s, g, MICRO_GROUP)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / float(qmax)
    q = (x / scale).round().clamp(-qmax - 1, qmax)
    return (q * scale).view(b, h, s, d)


def _kivi_grouped_token_len(seq_len: int, group_size: int, residual_length: int) -> int:
    if seq_len <= 0:
        return 0
    r = min(residual_length, seq_len)
    grouped_len = seq_len - r
    return max(0, (grouped_len // group_size) * group_size)


def _per_token_prefix_len(seq_len: int, residual_length: int) -> int:
    """Tokens quantized under per-token schemes; tail ``residual_length`` stays fp."""
    if seq_len <= 0:
        return 0
    r = min(max(residual_length, 0), seq_len)
    return seq_len - r


def _hadamard_sign_vector(
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = HADAMARD_SIGN_SEED,
) -> torch.Tensor:
    """QuaRot-style random ±1 diagonal, reproducible from ``seed``."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    bits = torch.randint(0, 2, (dim,), generator=g, dtype=torch.int64)
    return (bits * 2 - 1).to(device=device, dtype=dtype)


def fast_walsh_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Normalized FWHT along the last dim. Involutory: ``fwht(fwht(x)) == x``.

    Last dimension must be a power of two (head_dim=128 for Fast-dLLM v2 7B).
    Computed in fp32 then cast back so butterfly stages do not underflow fp16.
    """
    n = x.shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT requires power-of-two last dim, got {n}")
    rows = x.numel() // n
    y = x.reshape(rows, n).float().contiguous()
    h = 1
    while h < n:
        y = y.view(rows, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(rows, n)
        h *= 2
    y = y * (n ** -0.5)
    return y.view(x.shape).to(dtype=x.dtype)


def _quantize_per_token_asymmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    """One min/max scale per (batch, head, token) over full head_dim."""
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4, or 8, got {bits}")
    max_int = 2**bits - 1
    orig = x.dtype
    xf = x.float()
    mn = xf.amin(dim=-1, keepdim=True)
    mx = xf.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / float(max_int)).clamp(min=1e-8)
    q = ((xf - mn) / scale).round().clamp(0, max_int)
    return (q * scale + mn).to(dtype=orig)


def quantize_per_token_key_roundtrip(
    key: torch.Tensor,
    bits: int,
    *,
    residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
) -> torch.Tensor:
    """Naive per-token asymmetric key quant (no rotation). Tail stays fp."""
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4, or 8, got {bits}")
    t = key.shape[-2]
    prefix_len = _per_token_prefix_len(t, residual_length)
    if prefix_len <= 0:
        return key
    out = key.clone()
    out[..., :prefix_len, :] = _quantize_per_token_asymmetric(
        key[..., :prefix_len, :], bits
    )
    return out


def quantize_hadamard_per_token_roundtrip(
    key: torch.Tensor,
    bits: int,
    *,
    residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
    sign_seed: int = HADAMARD_SIGN_SEED,
    use_random_signs: bool = True,
) -> torch.Tensor:
    """Post-RoPE randomized Hadamard + per-token quant, returned in original space.

    ``k̂ = H⁻¹ dequant(quant(H (s ⊙ k)))`` with Walsh-Hadamard ``H`` (involutory)
    and optional QuaRot diagonal ``s ∈ {±1}^D``. Scores with the original query
    satisfy ``qᵀk̂ ≈ (H(s⊙q))ᵀ dequant(quant(H(s⊙k)))``. Tail stays fp16.
    """
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4, or 8, got {bits}")
    t = key.shape[-2]
    d = key.shape[-1]
    prefix_len = _per_token_prefix_len(t, residual_length)
    if prefix_len <= 0:
        return key

    out = key.clone()
    sl = key[..., :prefix_len, :]
    signs = None
    if use_random_signs:
        signs = _hadamard_sign_vector(d, sl.device, sl.dtype, sign_seed)
        sl = sl * signs
    sl_q = _quantize_per_token_asymmetric(fast_walsh_hadamard(sl), bits)
    sl_hat = fast_walsh_hadamard(sl_q)
    if signs is not None:
        sl_hat = sl_hat * signs
    out[..., :prefix_len, :] = sl_hat
    return out


def _rht(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return fast_walsh_hadamard(x * signs)


def _inv_rht(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return fast_walsh_hadamard(x) * signs


def quantize_quarot_per_token_roundtrip(
    key: torch.Tensor,
    bits: int,
    *,
    residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
    sign_seed: int = HADAMARD_SIGN_SEED,
) -> torch.Tensor:
    """Online QuaRot KV path, returned in original (post-RoPE) space.

    Paper: fuse a hidden-size randomized Hadamard ``Q`` into W_k, then apply
    head-wise ``I ⊗ H_{d_h}`` after RoPE. We cannot touch weights, so the
    online stand-in is: RHT over concatenated KV heads (``H_kv · d_h``, power
    of two for Fast-dLLM v2 = 4×128=512), then per-head RHT, then per-token
    quant. Inverse in reverse order. Falls back to per-head RHT if ``H·D``
    is not a power of two.
    """
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4, or 8, got {bits}")
    b, h, t, d = key.shape
    prefix_len = _per_token_prefix_len(t, residual_length)
    if prefix_len <= 0:
        return key
    n_flat = h * d
    if n_flat <= 0 or (n_flat & (n_flat - 1)) != 0:
        return quantize_hadamard_per_token_roundtrip(
            key, bits, residual_length=residual_length, sign_seed=sign_seed
        )

    out = key.clone()
    sl = key[..., :prefix_len, :]
    signs_a = _hadamard_sign_vector(n_flat, sl.device, sl.dtype, sign_seed)
    signs_b = _hadamard_sign_vector(d, sl.device, sl.dtype, sign_seed + 1)

    flat = sl.permute(0, 2, 1, 3).reshape(b, prefix_len, n_flat)
    mixed = _rht(flat, signs_a).view(b, prefix_len, h, d).permute(0, 2, 1, 3)
    rotated = _rht(mixed, signs_b)
    quantized = _quantize_per_token_asymmetric(rotated, bits)
    un_head = _inv_rht(quantized, signs_b)
    un_flat = un_head.permute(0, 2, 1, 3).reshape(b, prefix_len, n_flat)
    restored = _inv_rht(un_flat, signs_a).view(b, prefix_len, h, d).permute(0, 2, 1, 3)
    out[..., :prefix_len, :] = restored
    return out


def _qjl_jl_matrix(m: int, d: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randn(m, d, generator=g, dtype=torch.float32).to(device=device)


def quantize_qjl_key_roundtrip(
    key: torch.Tensor,
    bits: int,
    *,
    residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
    sign_seed: int = HADAMARD_SIGN_SEED,
    sketch_dim: int | None = None,
) -> torch.Tensor:
    """QJL (Zandieh et al., 2024): reconstruct k̂ with qᵀk̂ = Prod_QJL(q, k).

    ``H_S(k) = sign(S k)`` for ``S ∈ R^{m×d}`` i.i.d. N(0,1). Asymmetric
    estimator (quantize K only, leave Q in JL space):

        Prod_QJL(q, k) = √(π/2)/m · ‖k‖₂ · ⟨S q, sign(S k)⟩

    which is ``qᵀ k̂`` for ``k̂ = √(π/2)/m · ‖k‖₂ · Sᵀ sign(S k)``.
    ``bits`` sets sketch width ``m = d · bits`` unless ``sketch_dim`` is given
    (k2 → m=256, k4 → m=512 for head_dim=128). Tail stays fp.
    """
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4, or 8, got {bits}")
    t = key.shape[-2]
    d = key.shape[-1]
    prefix_len = _per_token_prefix_len(t, residual_length)
    if prefix_len <= 0:
        return key
    m = int(sketch_dim) if sketch_dim is not None else d * int(bits)
    if m <= 0:
        raise ValueError(f"QJL sketch_dim must be positive, got {m}")

    out = key.clone()
    sl = key[..., :prefix_len, :].float()
    S = _qjl_jl_matrix(m, d, sl.device, sign_seed)
    sketched = torch.matmul(sl, S.transpose(0, 1))
    signs = torch.where(sketched >= 0, torch.ones_like(sketched), -torch.ones_like(sketched))
    norms = sl.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = math.sqrt(math.pi / 2.0) / float(m)
    k_hat = scale * norms * torch.matmul(signs, S)
    out[..., :prefix_len, :] = k_hat.to(dtype=key.dtype)
    return out


def quantize_key_roundtrip(
    key: torch.Tensor,
    bits: int,
    *,
    scheme: str = KIVI_SCHEME,
    group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
    sign_seed: int = HADAMARD_SIGN_SEED,
    use_random_signs: bool = True,
) -> torch.Tensor:
    """Dispatch key quant-dequant: kivi | per_token | hadamard_per_token | quarot | qjl."""
    if scheme == KIVI_SCHEME:
        return quantize_kivi_key_roundtrip(
            key, bits, group_size=group_size, residual_length=residual_length
        )
    if scheme == PER_TOKEN_SCHEME:
        return quantize_per_token_key_roundtrip(
            key, bits, residual_length=residual_length
        )
    if scheme == HADAMARD_PER_TOKEN_SCHEME:
        return quantize_hadamard_per_token_roundtrip(
            key,
            bits,
            residual_length=residual_length,
            sign_seed=sign_seed,
            use_random_signs=use_random_signs,
        )
    if scheme == QUAROT_SCHEME:
        return quantize_quarot_per_token_roundtrip(
            key, bits, residual_length=residual_length, sign_seed=sign_seed
        )
    if scheme == QJL_SCHEME:
        return quantize_qjl_key_roundtrip(
            key, bits, residual_length=residual_length, sign_seed=sign_seed
        )
    raise ValueError(f"unknown key quant scheme {scheme!r}, expected one of {K_QUANT_SCHEMES}")


def quantize_value_roundtrip(
    value: torch.Tensor,
    bits: int,
    *,
    scheme: str = KIVI_SCHEME,
    residual_length: int = 0,
    sign_seed: int = HADAMARD_SIGN_SEED,
) -> torch.Tensor:
    """Value quant-dequant: kivi = per-token min/max; hadamard_per_token = RHT + per-token.

    Default residual_length=0 matches the existing V path (quantize all tokens).
    """
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4, or 8, got {bits}")
    if scheme == KIVI_SCHEME:
        t = value.shape[-2]
        prefix_len = _per_token_prefix_len(t, residual_length)
        if prefix_len <= 0:
            return value
        if prefix_len >= t:
            return _quantize_per_token_asymmetric(value, bits)
        out = value.clone()
        out[..., :prefix_len, :] = _quantize_per_token_asymmetric(
            value[..., :prefix_len, :], bits
        )
        return out
    if scheme == HADAMARD_PER_TOKEN_SCHEME:
        return quantize_hadamard_per_token_roundtrip(
            value,
            bits,
            residual_length=residual_length,
            sign_seed=sign_seed,
        )
    raise ValueError(
        f"unknown value quant scheme {scheme!r}, expected one of {V_QUANT_SCHEMES}"
    )


def quantize_kivi_key_roundtrip(
    key: torch.Tensor,
    bits: int,
    *,
    group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
) -> torch.Tensor:
    """KIVI key quant: asymmetric per-channel within each token group."""
    if bits not in (2, 4, 8):
        raise ValueError(f"KIVI bits must be 2, 4, or 8, got {bits}")
    b, h, t, d = key.shape
    grouped_len = _kivi_grouped_token_len(t, group_size, residual_length)
    if grouped_len == 0:
        return key

    max_int = 2**bits - 1
    out = key.clone()
    k_g = key[..., :grouped_len, :]
    num_groups = grouped_len // group_size
    x = k_g.view(b, h, num_groups, group_size, d)
    mn = x.amin(dim=-2, keepdim=True)
    mx = x.amax(dim=-2, keepdim=True)
    scale = ((mx - mn) / float(max_int)).clamp(min=1e-8)
    q = ((x - mn) / scale).round().clamp(0, max_int)
    out[..., :grouped_len, :] = (q * scale + mn).view(b, h, grouped_len, d)
    return out


def quantize_kivi_key_mixed_tiers(
    key: torch.Tensor,
    token_bits: list[int],
    *,
    group_size: int = DEFAULT_KIVI_GROUP_SIZE,
) -> torch.Tensor:
    """Gather-by-tier KIVI per-channel key quant across tokens; scatter back.

    ``token_bits[i]`` is the bitwidth for sequence position ``i`` (within ``key``'s
    token dimension). Positions with the same tier are gathered, quantized in groups
    of up to ``group_size`` tokens with per-channel min/max across the token dimension,
    then scattered back to original positions.
    """
    b, h, t, d = key.shape
    if len(token_bits) < t:
        raise ValueError(f"token_bits length {len(token_bits)} < key tokens {t}")

    out = key.clone()
    device = key.device
    for tier_bits in (4, 2):
        indices = [i for i in range(t) if token_bits[i] == tier_bits]
        if not indices:
            continue
        idx = torch.tensor(indices, device=device, dtype=torch.long)
        gathered = key.index_select(2, idx)
        n = gathered.shape[2]
        max_int = 2**tier_bits - 1
        deq_parts: list[torch.Tensor] = []
        for start in range(0, n, group_size):
            end = min(start + group_size, n)
            chunk = gathered[:, :, start:end, :]
            mn = chunk.amin(dim=-2, keepdim=True)
            mx = chunk.amax(dim=-2, keepdim=True)
            scale = ((mx - mn) / float(max_int)).clamp(min=1e-8)
            q = ((chunk - mn) / scale).round().clamp(0, max_int)
            deq_parts.append(q * scale + mn)
        deq_gathered = torch.cat(deq_parts, dim=2)
        out.index_copy_(2, idx, deq_gathered)
    return out


def _quantize_key_mixed_tiers_block(
    key: torch.Tensor,
    token_bits: list[int],
    grouped_len: int,
    *,
    keys_pre_rope: bool,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
) -> torch.Tensor:
    if grouped_len <= 0:
        return key
    bits_slice = token_bits[:grouped_len]
    k_slice = key[..., :grouped_len, :]
    if keys_pre_rope:
        assert cos is not None and sin is not None
        k_slice = inverse_rope_keys(k_slice, cos[:, :grouped_len, :], sin[:, :grouped_len, :])
        k_deq = quantize_kivi_key_mixed_tiers(
            k_slice, bits_slice, group_size=kivi_group_size
        )
        k_deq = apply_rope_keys(k_deq, cos[:, :grouped_len, :], sin[:, :grouped_len, :])
    else:
        k_deq = quantize_kivi_key_mixed_tiers(
            k_slice, bits_slice, group_size=kivi_group_size
        )
    out = key.clone()
    out[..., :grouped_len, :] = k_deq
    return out


def quantize_kivi_key_single_token(
    key: torch.Tensor,
    bits: int,
    *,
    group_size: int = DEFAULT_KIVI_GROUP_SIZE,
) -> torch.Tensor:
    """KIVI-style asymmetric key quant for one token [B, H, 1, D] (head_dim groups)."""
    if bits not in (2, 4, 8):
        raise ValueError(f"KIVI bits must be 2, 4, or 8, got {bits}")
    b, h, t, d = key.shape
    if t != 1:
        raise ValueError(f"expected one token (T=1), got T={t}")
    if d % group_size != 0:
        raise ValueError(f"head_dim {d} must be divisible by KIVI group_size {group_size}")

    max_int = 2**bits - 1
    num_groups = d // group_size
    x = key.view(b, h, 1, num_groups, group_size)
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / float(max_int)).clamp(min=1e-8)
    q = ((x - mn) / scale).round().clamp(0, max_int)
    return (q * scale + mn).view(b, h, 1, d)


def quantize_kivi_value_single_token(
    value: torch.Tensor,
    bits: int,
    *,
    group_size: int = DEFAULT_KIVI_GROUP_SIZE,
) -> torch.Tensor:
    """KIVI value quant for one token [B, H, 1, D] (head_dim groups)."""
    if bits not in (2, 4, 8):
        raise ValueError(f"KIVI bits must be 2, 4, or 8, got {bits}")
    b, h, t, d = value.shape
    if t != 1:
        raise ValueError(f"expected one token (T=1), got T={t}")
    if d % group_size != 0:
        raise ValueError(f"head_dim {d} must be divisible by KIVI group_size {group_size}")

    max_int = 2**bits - 1
    num_groups = d // group_size
    x = value.view(b, h, 1, num_groups, group_size)
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / float(max_int)).clamp(min=1e-8)
    q = ((x - mn) / scale).round().clamp(0, max_int)
    return (q * scale + mn).view(b, h, 1, d)


def quantize_kivi_value_roundtrip(
    value: torch.Tensor,
    bits: int,
    *,
    group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
) -> torch.Tensor:
    """KIVI value quant: asymmetric per-token over head_dim groups; tail stays fp."""
    if bits not in (2, 4, 8):
        raise ValueError(f"KIVI bits must be 2, 4, or 8, got {bits}")
    b, h, t, d = value.shape
    if d % group_size != 0:
        raise ValueError(f"head_dim {d} must be divisible by KIVI group_size {group_size}")

    grouped_len = _kivi_grouped_token_len(t, group_size, residual_length)
    if grouped_len == 0:
        return value

    max_int = 2**bits - 1
    out = value.clone()
    v_g = value[..., :grouped_len, :]
    num_groups = d // group_size
    x = v_g.view(b, h, grouped_len, num_groups, group_size)
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / float(max_int)).clamp(min=1e-8)
    q = ((x - mn) / scale).round().clamp(0, max_int)
    out[..., :grouped_len, :] = (q * scale + mn).view(b, h, grouped_len, d)
    return out


def quantize_tensor(
    tensor: torch.Tensor,
    block_size: int,
    num_blocks: int,
    bits: int,
    scheme: str = "block",
    *,
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
) -> torch.Tensor:
    """Round-trip quantize-dequant completed prefix; scheme: block | micro16 | kivi."""
    quant_len = num_blocks * block_size
    if num_blocks == 0:
        return tensor

    t = tensor[..., :quant_len, :]
    if scheme == "block":
        q, scale = quantize_per_block(t, block_size, num_blocks, bits=bits)
        deq = dequantize_per_block(q, scale, block_size)
    elif scheme == "micro16":
        blocks = t.view(t.shape[0], t.shape[1], num_blocks, block_size, t.shape[-1])
        deq_parts = [quantize_micro16(blocks[:, :, i], bits) for i in range(num_blocks)]
        deq = torch.cat(deq_parts, dim=2)
    elif scheme == KIVI_SCHEME:
        deq = quantize_kivi_value_roundtrip(
            t, bits, group_size=kivi_group_size, residual_length=kivi_residual_length
        )
    else:
        raise ValueError(f"unknown scheme {scheme!r}, expected block, micro16, or kivi")

    if quant_len < tensor.shape[-2]:
        return torch.cat([deq, tensor[..., quant_len:, :]], dim=-2)
    return deq


def quantize_int8_per_block(
    tensor: torch.Tensor, block_size: int, num_blocks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_per_block(tensor, block_size, num_blocks, bits=8)


def dequantize_int8_per_block(
    q: torch.Tensor, scale: torch.Tensor, block_size: int
) -> torch.Tensor:
    return dequantize_per_block(q, scale, block_size)


def _quantize_key_block(
    key: torch.Tensor,
    block_size: int,
    num_blocks: int,
    bits: int,
    scheme: str,
    *,
    keys_pre_rope: bool,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
) -> torch.Tensor:
    quant_len = num_blocks * block_size
    if scheme == KIVI_SCHEME:
        k_slice = key[..., :quant_len, :]
        if keys_pre_rope:
            assert cos is not None and sin is not None
            k_slice = inverse_rope_keys(k_slice, cos[:, :quant_len, :], sin[:, :quant_len, :])
            k_deq = quantize_kivi_key_roundtrip(
                k_slice, bits, group_size=kivi_group_size, residual_length=kivi_residual_length
            )
            k_deq = apply_rope_keys(k_deq, cos[:, :quant_len, :], sin[:, :quant_len, :])
        else:
            k_deq = quantize_kivi_key_roundtrip(
                k_slice, bits, group_size=kivi_group_size, residual_length=kivi_residual_length
            )
        if quant_len < key.shape[-2]:
            return torch.cat([k_deq, key[..., quant_len:, :]], dim=-2)
        return k_deq

    if keys_pre_rope:
        assert cos is not None and sin is not None
        k = inverse_rope_keys(key[..., :quant_len, :], cos[:, :quant_len, :], sin[:, :quant_len, :])
        k_deq = quantize_tensor(k, block_size, num_blocks, bits, scheme=scheme)
        k_deq = apply_rope_keys(k_deq, cos[:, :quant_len, :], sin[:, :quant_len, :])
    else:
        k_deq = quantize_tensor(key, block_size, num_blocks, bits, scheme=scheme)

    if quant_len < key.shape[-2]:
        return torch.cat([k_deq, key[..., quant_len:, :]], dim=-2)
    return k_deq


def quantize_completed_blocks(
    past_key_values: DynamicCache | None,
    block_size: int,
    bits: int = 8,
    *,
    scheme: str = "block",
    keys_pre_rope: bool = False,
    model=None,
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
) -> DynamicCache | None:
    """Round-trip completed blocks through int4/int8 quantization."""
    if past_key_values is None or block_size <= 0 or bits <= 0:
        return past_key_values

    cos = sin = None
    if keys_pre_rope:
        if model is None:
            raise ValueError("model is required when keys_pre_rope=True")
        key0 = past_key_values.key_cache[0]
        cos, sin = rope_cos_sin(
            model,
            key0.shape[-2],
            key0.shape[0],
            key0.device,
            key0.dtype,
        )

    for layer_id in range(len(past_key_values)):
        key = past_key_values.key_cache[layer_id]
        value = past_key_values.value_cache[layer_id]
        seq_len = key.shape[-2]
        num_blocks = seq_len // block_size
        if num_blocks == 0:
            continue

        past_key_values.key_cache[layer_id] = _quantize_key_block(
            key,
            block_size,
            num_blocks,
            bits,
            scheme,
            keys_pre_rope=keys_pre_rope,
            cos=cos,
            sin=sin,
            kivi_group_size=kivi_group_size,
            kivi_residual_length=kivi_residual_length,
        )
        if scheme == KIVI_SCHEME:
            v_slice = value[..., : num_blocks * block_size, :]
            v_deq = quantize_kivi_value_roundtrip(
                v_slice, bits, group_size=kivi_group_size, residual_length=kivi_residual_length
            )
            if num_blocks * block_size < value.shape[-2]:
                past_key_values.value_cache[layer_id] = torch.cat(
                    [v_deq, value[..., num_blocks * block_size :, :]], dim=-2
                )
            else:
                past_key_values.value_cache[layer_id] = v_deq
        else:
            past_key_values.value_cache[layer_id] = quantize_tensor(
                value,
                block_size,
                num_blocks,
                bits,
                scheme=scheme,
                kivi_group_size=kivi_group_size,
                kivi_residual_length=kivi_residual_length,
            )

    return past_key_values


def quantize_completed_blocks_kivi_split(
    past_key_values: DynamicCache | None,
    block_size: int,
    *,
    key_bits: int,
    value_bits: int,
    keys_pre_rope: bool = False,
    model=None,
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
) -> DynamicCache | None:
    """Uniform KIVI quant with independent key/value bitwidth."""
    if past_key_values is None or block_size <= 0:
        return past_key_values

    cos = sin = None
    if keys_pre_rope:
        if model is None:
            raise ValueError("model is required when keys_pre_rope=True")
        key0 = past_key_values.key_cache[0]
        cos, sin = rope_cos_sin(
            model,
            key0.shape[-2],
            key0.shape[0],
            key0.device,
            key0.dtype,
        )

    for layer_id in range(len(past_key_values)):
        key = past_key_values.key_cache[layer_id]
        value = past_key_values.value_cache[layer_id]
        seq_len = key.shape[-2]
        num_blocks = seq_len // block_size
        if num_blocks == 0:
            continue

        past_key_values.key_cache[layer_id] = _quantize_key_block(
            key,
            block_size,
            num_blocks,
            key_bits,
            KIVI_SCHEME,
            keys_pre_rope=keys_pre_rope,
            cos=cos,
            sin=sin,
            kivi_group_size=kivi_group_size,
            kivi_residual_length=kivi_residual_length,
        )
        v_slice = value[..., : num_blocks * block_size, :]
        v_deq = quantize_kivi_value_roundtrip(
            v_slice,
            value_bits,
            group_size=kivi_group_size,
            residual_length=kivi_residual_length,
        )
        if num_blocks * block_size < value.shape[-2]:
            past_key_values.value_cache[layer_id] = torch.cat(
                [v_deq, value[..., num_blocks * block_size :, :]], dim=-2
            )
        else:
            past_key_values.value_cache[layer_id] = v_deq

    return past_key_values


def apply_adaptive_k_mixed_quant(
    past_key_values: DynamicCache | None,
    k_bits: list[int],
    v_bits_per_layer: dict[int, list[int]],
    *,
    num_cached: int,
    block_size: int,
    keys_pre_rope: bool = False,
    model=None,
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
) -> DynamicCache | None:
    """Mixed-tier KIVI keys (gather-by-tier) + per-token adaptive V4/V2."""
    if past_key_values is None or num_cached <= 0:
        return past_key_values

    seq_len = past_key_values.key_cache[0].shape[-2]
    num_blocks = seq_len // block_size
    if num_blocks == 0:
        return past_key_values

    cos = sin = None
    if keys_pre_rope:
        if model is None:
            raise ValueError("model is required when keys_pre_rope=True")
        key0 = past_key_values.key_cache[0]
        cos, sin = rope_cos_sin(
            model,
            key0.shape[-2],
            key0.shape[0],
            key0.device,
            key0.dtype,
        )

    grouped_len = _kivi_grouped_token_len(seq_len, kivi_group_size, kivi_residual_length)
    n_apply = min(num_cached, num_blocks * block_size)
    k_apply = min(len(k_bits), grouped_len, n_apply)

    for layer_id in range(len(past_key_values)):
        key = past_key_values.key_cache[layer_id]
        if k_apply > 0:
            past_key_values.key_cache[layer_id] = _quantize_key_mixed_tiers_block(
                key,
                k_bits,
                k_apply,
                keys_pre_rope=keys_pre_rope,
                cos=cos,
                sin=sin,
                kivi_group_size=kivi_group_size,
            )

        value = past_key_values.value_cache[layer_id]
        v_slice = value[..., : num_blocks * block_size, :]
        v_deq = quantize_kivi_value_roundtrip(
            v_slice,
            4,
            group_size=kivi_group_size,
            residual_length=kivi_residual_length,
        )
        if num_blocks * block_size < value.shape[-2]:
            past_key_values.value_cache[layer_id] = torch.cat(
                [v_deq, value[..., num_blocks * block_size :, :]], dim=-2
            )
        else:
            past_key_values.value_cache[layer_id] = v_deq

        v_bits = v_bits_per_layer.get(layer_id)
        if not v_bits:
            continue
        value = past_key_values.value_cache[layer_id]
        n_v = min(len(v_bits), n_apply)
        for tok in range(n_v):
            v_tok = value[:, :, tok : tok + 1, :]
            value[:, :, tok : tok + 1, :] = quantize_kivi_value_single_token(
                v_tok,
                v_bits[tok],
                group_size=kivi_group_size,
            )

    return past_key_values


def apply_adaptive_v_kivi_quant(
    past_key_values: DynamicCache | None,
    v_bits_per_layer: dict[int, list[int]],
    *,
    num_cached: int,
    block_size: int,
    key_bits: int = 2,
    keys_pre_rope: bool = False,
    model=None,
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
) -> DynamicCache | None:
    """Keep uniform KIVI keys; apply per-token mixed V4/V2 on cached prefix."""
    if past_key_values is None or num_cached <= 0:
        return past_key_values

    seq_len = past_key_values.key_cache[0].shape[-2]
    num_blocks = seq_len // block_size
    if num_blocks == 0:
        return past_key_values

    quantize_completed_blocks_kivi_split(
        past_key_values,
        block_size,
        key_bits=key_bits,
        value_bits=4,
        keys_pre_rope=keys_pre_rope,
        model=model,
        kivi_group_size=kivi_group_size,
        kivi_residual_length=kivi_residual_length,
    )

    n_apply = min(num_cached, num_blocks * block_size)
    for layer_id in range(len(past_key_values)):
        bits = v_bits_per_layer.get(layer_id)
        if not bits:
            continue
        value = past_key_values.value_cache[layer_id]
        n = min(len(bits), n_apply)
        for tok in range(n):
            v_tok = value[:, :, tok : tok + 1, :]
            value[:, :, tok : tok + 1, :] = quantize_kivi_value_single_token(
                v_tok,
                bits[tok],
                group_size=kivi_group_size,
            )

    return past_key_values
