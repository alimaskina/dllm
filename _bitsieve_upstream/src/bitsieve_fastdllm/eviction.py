"""Bounded KV cache: keep a live set instead of the whole prefix.

Selection already decides what a *block* reads. Eviction decides what the cache
*keeps*, which is a different question: a prefix entry no selector has looked at
for twenty blocks is paying for itself every block it stays.

State, per (layer, KV head), one slot per live entry:

    g    fp32    EMA of the attention mass the entry has been receiving
    n    uint16  age, in blocks, since the entry entered the cache
    pos  int32   the entry's original position in the cache

At step 0 of every block the selector already computes ``alpha`` -- the softmax
mass each prefix entry receives from the block's representative queries -- so
the update costs nothing beyond the arithmetic:

    g[i]  <- lam * g[i] + (1 - lam) * alpha[i]
    n[i]  <- n[i] + 1
    ghat  =  g[i] / (1 - lam ** n[i])

The ``1 - lam**n`` correction is not optional. ``g`` starts at zero, so without
it a brand-new entry reads as near-worthless purely because it has been
observed once, and any policy with a recency window would spend that window
undoing the artifact rather than protecting genuinely useful new tokens. With
the correction a first observation yields exactly ``ghat = alpha``.

Every ``interval_blocks`` blocks, if the live set exceeds
``C = max(capacity_percent% of tokens seen, capacity_floor)``:

    recent      keep the C newest by ``pos``
    ema_recent  keep the W newest by ``pos``, plus the top ``C - W`` by ``ghat``
                among the rest

Eviction compacts the state arrays; it is not a mask. Evicted entries leave the
candidate set entirely, so they are not ranked, not read, and not counted --
see ``docs/eviction.md`` for what "not counted" means against a cache whose
buffers are preallocated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

Policy = Literal["none", "recent", "ema_recent", "ema_recent_value"]


@dataclass(slots=True)
class EvictionConfig:
    policy: Policy = "none"
    decay: float = 0.9              # lambda
    recent_window: int = 128        # W
    capacity_percent: float = 5.0
    capacity_floor: int = 256
    interval_blocks: int = 4

    @property
    def enabled(self) -> bool:
        return self.policy != "none"

    def validate(self) -> None:
        if self.policy not in ("none", "recent", "ema_recent", "ema_recent_value"):
            raise ValueError(f"unknown eviction policy: {self.policy}")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {self.decay}")
        if self.recent_window < 0:
            raise ValueError("recent_window must be non-negative")
        if self.capacity_floor <= 0:
            raise ValueError("capacity_floor must be positive")
        if not 0.0 < self.capacity_percent <= 100.0:
            raise ValueError("capacity_percent must be in (0, 100]")
        if self.interval_blocks <= 0:
            raise ValueError("interval_blocks must be positive")
        if self.policy.startswith("ema_recent") and self.recent_window > self.capacity_floor:
            raise ValueError(
                f"recent_window={self.recent_window} exceeds capacity_floor="
                f"{self.capacity_floor}; the window alone would fill the budget "
                "and no entry could ever be kept on its EMA"
            )

    def capacity(self, tokens_seen: int) -> int:
        """C = max(capacity_percent% of everything the cache has held, floor)."""
        return max(int(self.capacity_percent / 100.0 * tokens_seen), self.capacity_floor)


class EvictionState:
    """Per (layer, batch, KV head) live-entry state.

    Every (layer, head) keeps its own live set -- different heads attend to
    different things, so forcing one shared set would make the policy as weak as
    its least selective head. The sets stay the same *size*, which is what lets
    the cache keep one length.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        capacity: int,
        device: torch.device | str,
        max_position: int = 1 << 31,
    ) -> None:
        self.num_layers = int(num_layers)
        self.batch_size = int(batch_size)
        self.num_kv_heads = int(num_kv_heads)
        self.device = torch.device(device)
        # int32 addresses any cache this runtime can build; int64 only when a
        # caller genuinely needs it, since pos is a third of the state's size.
        self.pos_dtype = torch.int32 if max_position < (1 << 31) else torch.int64

        shape = (self.num_layers, self.batch_size, self.num_kv_heads, capacity)
        self.g = torch.zeros(shape, dtype=torch.float32, device=self.device)
        # int16, not uint16: the spec's width, but a signed type because torch's
        # uint16 supports almost no arithmetic. 4096 tokens / 32 = 128 blocks, so
        # the range is never in question.
        self.n = torch.zeros(shape, dtype=torch.int16, device=self.device)
        self.pos = torch.zeros(shape, dtype=self.pos_dtype, device=self.device)
        self.live = 0
        self._capacity = capacity

    # -- growth ------------------------------------------------------------
    def _grow_to(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        new_cap = max(needed, self._capacity * 2, 1)
        pad = new_cap - self._capacity
        shape = (self.num_layers, self.batch_size, self.num_kv_heads, pad)
        self.g = torch.cat([self.g, torch.zeros(shape, dtype=self.g.dtype, device=self.device)], -1)
        self.n = torch.cat([self.n, torch.zeros(shape, dtype=self.n.dtype, device=self.device)], -1)
        self.pos = torch.cat(
            [self.pos, torch.zeros(shape, dtype=self.pos.dtype, device=self.device)], -1
        )
        self._capacity = new_cap

    def append(self, positions: torch.Tensor | range | list[int]) -> None:
        """Admit newly committed cache entries with g = 0, n = 0."""
        if isinstance(positions, range):
            positions = list(positions)
        if isinstance(positions, list):
            positions = torch.tensor(positions, dtype=self.pos_dtype, device=self.device)
        m = int(positions.numel())
        if m == 0:
            return
        self._grow_to(self.live + m)
        sl = slice(self.live, self.live + m)
        self.g[..., sl] = 0.0
        self.n[..., sl] = 0
        self.pos[..., sl] = positions.to(self.pos_dtype).view(1, 1, 1, m)
        self.live += m

    # -- update ------------------------------------------------------------
    def observe(self, layer_idx: int, alpha: torch.Tensor, decay: float) -> None:
        """Fold this block's attention mass into the EMA for one layer.

        ``alpha``: [B, Hkv, live] -- the selector's per-entry softmax mass,
        already aligned with the live set because the candidate set *is* the
        live set.
        """
        if self.live == 0:
            return
        if alpha.shape[-1] != self.live:
            raise ValueError(
                f"alpha has {alpha.shape[-1]} entries but {self.live} are live; "
                "the selector must rank exactly the live set"
            )
        sl = slice(0, self.live)
        g = self.g[layer_idx, ..., sl]
        g.mul_(decay).add_(alpha.to(g.dtype), alpha=1.0 - decay)
        self.n[layer_idx, ..., sl] += 1

    def corrected(self, layer_idx: int, decay: float) -> torch.Tensor:
        """ghat = g / (1 - lam**n), the bias-corrected EMA. [B, Hkv, live]."""
        sl = slice(0, self.live)
        g = self.g[layer_idx, ..., sl]
        n = self.n[layer_idx, ..., sl]
        if decay == 0.0:
            return g.clone()
        denom = 1.0 - torch.pow(
            torch.tensor(decay, dtype=torch.float32, device=g.device), n.to(torch.float32)
        )
        # n == 0 means never observed: leave it at zero rather than dividing by
        # zero, so an entry admitted this block is not ranked on noise.
        return torch.where(n > 0, g / denom.clamp_min(1e-8), torch.zeros_like(g))

    # -- eviction ----------------------------------------------------------
    def survivors(
        self,
        cfg: EvictionConfig,
        capacity: int,
        value_spread: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Indices into the live axis to keep. [L, B, Hkv, capacity], sorted by pos.

        ``value_spread``: [L, B, Hkv, live] -- ||v_i - v_head_mean|| for every
        live entry, required by ``ema_recent_value`` and ignored otherwise. The
        EMA contest then ranks on ghat * spread rather than ghat alone: an entry
        is worth keeping only if it is both attended to *and* says something the
        head's average value does not already say.
        """
        sl = slice(0, self.live)
        pos = self.pos[..., sl]

        if cfg.policy == "recent":
            keep = pos.topk(capacity, dim=-1).indices
        elif cfg.policy in ("ema_recent", "ema_recent_value"):
            w = min(cfg.recent_window, capacity)
            recent = pos.topk(w, dim=-1).indices if w else None
            rest = capacity - w
            if rest > 0:
                ghat = torch.stack(
                    [self.corrected(l, cfg.decay) for l in range(self.num_layers)], dim=0
                )
                if cfg.policy == "ema_recent_value":
                    if value_spread is None:
                        raise ValueError(
                            "policy ema_recent_value needs value_spread; the caller "
                            "must supply per-entry ||v - vbar||"
                        )
                    if value_spread.shape != ghat.shape:
                        raise ValueError(
                            f"value_spread has shape {tuple(value_spread.shape)} but "
                            f"the live EMA is {tuple(ghat.shape)}"
                        )
                    ghat = ghat * value_spread.to(ghat.dtype)
                if recent is not None:
                    # Exclude the window from the EMA contest; an entry must not
                    # be able to win a slot it already holds.
                    ghat = ghat.scatter(-1, recent, float("-inf"))
                extra = ghat.topk(rest, dim=-1).indices
                keep = torch.cat([recent, extra], dim=-1) if recent is not None else extra
            else:
                keep = recent
        else:
            raise ValueError(f"policy {cfg.policy} does not evict")

        # Keep the live arrays ordered by position so "recent" stays the tail.
        keep_pos = pos.gather(-1, keep)
        return keep.gather(-1, keep_pos.argsort(dim=-1))

    def compact(self, keep: torch.Tensor) -> None:
        """Physically drop everything outside ``keep``."""
        c = keep.shape[-1]
        self.g[..., :c] = self.g[..., : self.live].gather(-1, keep)
        self.n[..., :c] = self.n[..., : self.live].gather(-1, keep)
        self.pos[..., :c] = self.pos[..., : self.live].gather(-1, keep.to(torch.int64))
        self.live = c

    def maybe_evict(
        self,
        cfg: EvictionConfig,
        *,
        block_index: int,
        tokens_seen: int,
        value_spread_fn: "Callable[[], torch.Tensor] | None" = None,
    ) -> torch.Tensor | None:
        """Evict on schedule. Returns the survivor indices, or None if nothing ran."""
        if not cfg.enabled or self.live == 0:
            return None
        if (block_index + 1) % cfg.interval_blocks:
            return None
        capacity = cfg.capacity(tokens_seen)
        if self.live <= capacity:
            return None
        spread = None
        if cfg.policy == "ema_recent_value":
            if value_spread_fn is None:
                raise ValueError("policy ema_recent_value needs a value_spread_fn")
            spread = value_spread_fn()
        keep = self.survivors(cfg, capacity, spread)
        self.compact(keep)
        return keep

    # -- accounting --------------------------------------------------------
    def state_nbytes(self) -> int:
        """Bytes the policy itself costs. The spec says to count it, so count it."""
        per_entry = (
            self.g.element_size() + self.n.element_size() + self.pos.element_size()
        )
        return self.num_layers * self.batch_size * self.num_kv_heads * self.live * per_entry


def live_cache_nbytes(
    pos: torch.Tensor,
    *,
    live: int,
    head_dim: int,
    k_bits: int,
    v_bits: int,
    key_token_group: int,
    value_channel_group: int,
    param_bytes: int = 2,
    compute_bytes: int = 2,
) -> int:
    """Bytes a cache holding exactly this live set would occupy.

    ``pos``: [L, B, Hkv, capacity] -- the live positions, of which the first
    ``live`` are valid.

    Keys are quantized per 32-token *group*, so an entry carries a reference to
    its group's scale vector rather than a scale of its own. A group therefore
    survives as long as any one of its entries does, and the metadata is counted
    per surviving group, not per surviving entry -- which is why a policy that
    keeps a contiguous tail is cheaper per entry than one that keeps a scatter.
    """
    if live <= 0:
        return 0
    l, b, h = pos.shape[0], pos.shape[1], pos.shape[2]
    live_pos = pos[..., :live]

    if k_bits >= 16:
        k_payload = l * b * h * live * head_dim * compute_bytes
        k_meta = 0
    else:
        k_payload = l * b * h * live * head_dim * k_bits // 8
        groups = torch.unique(live_pos // key_token_group, dim=-1)
        # unique() pads with repeats per row, so count distinct values per row.
        n_groups = 0
        flat = (live_pos // key_token_group).reshape(-1, live)
        for row in flat:
            n_groups += int(torch.unique(row).numel())
        k_meta = n_groups * head_dim * 2 * param_bytes

    if v_bits >= 16:
        v_payload = l * b * h * live * head_dim * compute_bytes
        v_meta = 0
    else:
        v_payload = l * b * h * live * head_dim * v_bits // 8
        v_meta = l * b * h * live * (head_dim // value_channel_group) * 2 * param_bytes

    return k_payload + k_meta + v_payload + v_meta
