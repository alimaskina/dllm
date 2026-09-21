"""BitSieve decoding for DreamReasoner.

Dream's own ``block_diffusion_generate`` already has the shape the session
expects -- prefill the block-aligned prompt, denoise a block against the cache,
commit it with one ``store_kv=True`` forward -- so this drives that loop with a
:class:`PackedKVCache` underneath instead of a ``DynamicCache``, and reuses
Dream's sampling and transfer rules verbatim rather than restating them.

Prefill runs with the patch *off* and a normal DynamicCache, then packs the
result, exactly as the Fast-dLLM generator does: the prompt is encoded once at
full precision and only then quantized, which is what the commit path does for
every later block too.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from ..cache import PackedKVCache
from ..config import ExperimentConfig
from .dream_adapter import DreamPatchHandle, patch_dream
from .session import BitSieveSession
from .trace import RunTrace


@dataclass
class DreamGenerationResult:
    sequences: torch.Tensor
    generated_ids: torch.Tensor
    texts: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    trace: RunTrace | None = None


def _gen_utils(model: torch.nn.Module):
    """Dream's own generation helpers, from whichever remote-code module holds them."""
    for base in type(model).__mro__:
        mod = sys.modules.get(base.__module__)
        if mod is not None and hasattr(mod, "build_block_diffusion_attention_mask"):
            return mod
    raise RuntimeError(
        "could not locate Dream's generation utilities; is this a Dream checkpoint?"
    )


class DreamBitSieveGenerator:

    def __init__(self, model: torch.nn.Module, tokenizer, config: ExperimentConfig) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        cfg = model.config
        self.num_layers = int(cfg.num_hidden_layers)
        self.num_q_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(
            getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        )
        self.device = next(model.parameters()).device
        self.compute_dtype = next(model.parameters()).dtype
        self.config.validate(head_dim=self.head_dim, num_layers=self.num_layers)
        self.gu = _gen_utils(model)
        self.mask_token_id = int(getattr(cfg, "mask_token_id", config.generation.mask_token_id))
        # The checkpoint's own EOS, not the experiment config's -- that default
        # is Fast-dLLM's 151645, and with the wrong id nothing ever stops: every
        # request runs to the full token budget, burning compute and letting the
        # model ramble past an answer it had already written.
        self.stop_token_id = getattr(cfg, "eos_token_id", None)
        if self.stop_token_id is None:
            self.stop_token_id = config.generation.stop_token_id
        self.patch: DreamPatchHandle = patch_dream(model, None)

    def _new_cache(self, batch_size: int) -> PackedKVCache:
        return PackedKVCache(
            num_layers=self.num_layers,
            batch_size=batch_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_tokens=self.config.max_cache_tokens,
            quant=self.config.quant,
            device=self.device,
            compute_dtype=self.compute_dtype,
            backend=self.config.backend,
        )

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> DreamGenerationResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must be [1, sequence]; batching is not wired")
        gu = self.gu
        cfg = self.config.generation
        block = cfg.block_size
        torch.manual_seed(self.config.seed)

        x0 = input_ids.to(self.device)
        prompt_len = int(x0.shape[1])
        gen_length = cfg.max_new_tokens
        num_blocks = (prompt_len + gen_length + block - 1) // block
        total = num_blocks * block
        attn_mask = gu.build_block_diffusion_attention_mask(
            num_blocks, block, self.device, batch_size=1
        )
        position_ids = torch.arange(total, device=self.device).unsqueeze(0)

        x = torch.full((1, total), self.mask_token_id, dtype=x0.dtype, device=self.device)
        x[:, :prompt_len] = x0

        prefill_blocks = prompt_len // block
        prefill_len = prefill_blocks * block
        cache = self._new_cache(1)
        prefill_keys: dict[int, torch.Tensor] = {}

        # ---- prefill, unpatched and at full precision ----------------------
        if prefill_len > 0:
            self.patch.set_session(None)
            dyn = DynamicCache()
            self.model(
                x[:, :prefill_len],
                attention_mask=attn_mask[:, :prefill_len, :prefill_len],
                position_ids=position_ids[:, :prefill_len],
                past_key_values=dyn,
                use_cache=True,
                store_kv=True,
            )
            cache.load_dynamic_cache(dyn)
            # Snapshot the prefill's exact keys before the fp16 cache is dropped:
            # they are the reference the packed selection gets scored against.
            # Without this the coverage column comes back empty and says nothing
            # about the selection, which is a silent hole rather than an error.
            prefill_keys = {}
            if self.config.coverage_diagnostics:
                for layer_idx in range(self.num_layers):
                    try:
                        key, _ = dyn[layer_idx]
                    except Exception:
                        if hasattr(dyn, "key_cache"):
                            key = dyn.key_cache[layer_idx]
                        else:
                            key = dyn.layers[layer_idx].keys
                    prefill_keys[layer_idx] = key[..., :prefill_len, :].detach().clone()
            del dyn

        session = BitSieveSession(
            cache, self.config,
            num_q_heads=self.num_q_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, compute_dtype=self.compute_dtype,
        )
        for layer_idx, key in prefill_keys.items():
            session.seed_fp16_key_shadow(layer_idx, key)
        prefill_keys.clear()
        self.patch.set_session(session)

        steps = cfg.fixed_steps_per_block if cfg.schedule == "fixed" else block
        transfer = gu.get_num_transfer_tokens(block, steps)
        nfe = 1 if prefill_len > 0 else 0
        stop_ids = gu._resolve_stopping_ids(self.stop_token_id)

        try:
            for nb in range(prefill_blocks, num_blocks):
                bs_, be_ = nb * block, (nb + 1) * block
                cur = x[:, bs_:be_].clone()
                session.begin_block(
                    (cur[0] == self.mask_token_id).nonzero(as_tuple=True)[0].tolist()
                )
                step = 0
                for step in range(steps + 1):
                    mask_index = cur == self.mask_token_id
                    if not bool(mask_index.any()):
                        break
                    session.prepare_forward(
                        step,
                        masked_positions=mask_index[0].nonzero(as_tuple=True)[0].tolist(),
                        commit_current=False,
                    )
                    logits = self.model(
                        cur,
                        attention_mask=attn_mask[:, bs_:be_, :be_],
                        position_ids=position_ids[:, bs_:be_],
                        past_key_values=cache,
                        use_cache=True,
                        store_kv=False,
                    ).logits
                    nfe += 1
                    cand, conf = gu.sample_with_temperature_topk_topp(
                        logits, temperature=cfg.temperature, top_k=0, top_p=cfg.top_p
                    )
                    cand = torch.where(mask_index, cand, cur)
                    take = gu._select_transfer_index(
                        "low_confidence_dynamic", mask_index, cand, conf, transfer, step,
                        cfg.threshold, 0.35, force_accept=(step == steps - 1),
                    )
                    cur[take] = cand[take]
                    session.finish_step0()

                if bool((cur == self.mask_token_id).any()):
                    raise RuntimeError("denoising ended with unresolved mask tokens")

                # ---- commit the finished block ------------------------------
                cache.begin_append(block)
                try:
                    session.prepare_forward(step, masked_positions=[], commit_current=True)
                    self.model(
                        cur,
                        attention_mask=attn_mask[:, bs_:be_, :be_],
                        position_ids=position_ids[:, bs_:be_],
                        past_key_values=cache,
                        use_cache=True,
                        store_kv=True,
                    )
                    cache.commit_append()
                except Exception:
                    cache.abort_append()
                    raise
                nfe += 1
                session.end_block()

                x[:, bs_:be_] = cur
                if gu._should_stop(x, prompt_len, stop_ids):
                    break
        finally:
            self.patch.set_session(None)

        out_len = min(total, prompt_len + gen_length)
        seq = x[:, :out_len]
        generated = seq[:, prompt_len:]
        trace = session.finalize_trace()
        metrics: dict[str, Any] = {
            "coverage": trace.coverage_summary(),
            "nfe": nfe,
            "cache_tokens": cache.get_seq_length(),
            "generated_tokens_per_request": int(generated.shape[1]),
            "blocks_sparse": session.trace.counters.get("blocks_sparse", 0),
            "blocks_dense_bypass": session.trace.counters.get("blocks_dense_bypass", 0),
            **cache.logical_nbytes(),
            **session.eviction_metrics(),
        }
        return DreamGenerationResult(
            sequences=seq,
            generated_ids=generated,
            texts=self.tokenizer.batch_decode(generated, skip_special_tokens=True),
            metrics=metrics,
            trace=trace,
        )
