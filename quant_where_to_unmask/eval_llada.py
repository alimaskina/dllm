#!/usr/bin/env python3
"""
LLaDA lm-eval harness wrapper (from ML-GSAI/LLaDA eval_llada.py).

Extended with quantization support matching quant_unmask:
  quant=fp16 | bf16 | int8 | int4
"""
import accelerate
import gzip
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from generate import generate


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_llada_model(model_path, quant, model_kwargs):
    """Load LLaDA with the same quant settings as quant_unmask."""
    kwargs = {"trust_remote_code": True, **model_kwargs}
    if quant in ("fp16", "bf16"):
        kwargs["torch_dtype"] = torch.float16 if quant == "fp16" else torch.bfloat16
    elif quant == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        if "device_map" not in kwargs:
            kwargs["device_map"] = "auto"
    elif quant == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if "device_map" not in kwargs:
            kwargs["device_map"] = "auto"
    else:
        raise ValueError(f"Unknown quant: {quant}")
    return AutoModel.from_pretrained(model_path, **kwargs)


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path="",
        quant="bf16",
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.0,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking="low_confidence",
        device="cuda",
        checkpoint_dir=None,
        gen_batch_size=8,
        **kwargs,
    ):
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})

        self.model = load_llada_model(model_path, quant, model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            if quant in ("fp16", "bf16"):
                self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            if quant in ("fp16", "bf16"):
                self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.quant = quant

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.0
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.checkpoint_dir = checkpoint_dir
        self.gen_batch_size = max(1, int(gen_batch_size))
        self._pad_id = self.tokenizer.pad_token_id
        if self._pad_id is None:
            self._pad_id = self.tokenizer.eos_token_id or 0
        self._trace_executor: ThreadPoolExecutor | None = None
        if checkpoint_dir:
            self._trace_executor = ThreadPoolExecutor(max_workers=2)

    def _collate_prompts(self, prompt_tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Left-pad prompts to common length; return (padded, attention_mask, raw_lens)."""
        raw_lens = [int(p.shape[0]) for p in prompt_tensors]
        max_plen = max(raw_lens)
        batch = torch.full((len(prompt_tensors), max_plen), self._pad_id, dtype=torch.long)
        attn = torch.zeros((len(prompt_tensors), max_plen), dtype=torch.long)
        for i, (p, plen) in enumerate(zip(prompt_tensors, raw_lens)):
            batch[i, max_plen - plen:] = p
            attn[i, max_plen - plen:] = 1
        return batch, attn, raw_lens

    def _checkpoint_file(self) -> Optional[Path]:
        if not self.checkpoint_dir:
            return None
        p = Path(self.checkpoint_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / f"rank{self.rank}.jsonl"

    def _trace_dir(self) -> Optional[Path]:
        if not self.checkpoint_dir:
            return None
        p = Path(self.checkpoint_dir) / "traces" / f"rank{self.rank}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _trace_file(self, idx: int) -> Optional[Path]:
        d = self._trace_dir()
        if d is None:
            return None
        return d / f"{idx}.json.gz"

    @staticmethod
    def _write_trace_file(path: Path, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wb", compresslevel=1) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def _flush_trace_writes(self) -> None:
        if self._trace_executor is not None:
            self._trace_executor.shutdown(wait=True)
            self._trace_executor = ThreadPoolExecutor(max_workers=2)

    def _load_checkpoint(self) -> dict[int, str]:
        path = self._checkpoint_file()
        done: dict[int, str] = {}
        if path is None or not path.exists():
            return done
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                done[int(row["idx"])] = row["response"]
        return done

    def _save_trace(
        self,
        idx: int,
        doc_id,
        prompt_text: str,
        trace: list,
        final_completion: str,
        *,
        gold_answer: Optional[str] = None,
    ) -> Optional[str]:
        path = self._trace_file(idx)
        if path is None:
            return None
        payload = {
            "idx": idx,
            "doc_id": doc_id,
            "rank": self.rank,
            "quant": self.quant,
            "gen_length": self.gen_length,
            "steps": self.steps,
            "block_length": self.block_length,
            "remasking": self.remasking,
            "mask_id": self.mask_id,
            "prompt_text": prompt_text,
            "gold_answer": gold_answer,
            "final_completion": final_completion,
            "steps_trace": trace,
        }
        if self._trace_executor is not None:
            self._trace_executor.submit(self._write_trace_file, path, payload)
        else:
            self._write_trace_file(path, payload)
        return str(path.relative_to(Path(self.checkpoint_dir)))

    def _append_checkpoint(
        self,
        idx: int,
        doc_id,
        response: str,
        *,
        prompt_text: str = "",
        trace_path: Optional[str] = None,
    ) -> None:
        path = self._checkpoint_file()
        if path is None:
            return
        row = {
            "idx": idx,
            "doc_id": doc_id,
            "response": response,
            "prompt_text": prompt_text,
            "trace_path": trace_path,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction="none") / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix

        for _i in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
        return torch.all(correct)

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
                torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]["until"]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        done = self._load_checkpoint()
        if done:
            print(f"[rank {self.rank}] Resuming from checkpoint: {len(done)}/{len(requests)} done")

        out: list[str | None] = [None] * len(requests)
        for i in done:
            if i < len(requests):
                out[i] = done[i]

        pending = [i for i in range(len(requests)) if out[i] is None]
        pbar = tqdm(total=len(requests), desc=f"Generating ({self.quant}) rank={self.rank}", initial=len(done))

        record_trace = self.checkpoint_dir is not None
        batch_size = self.gen_batch_size

        for batch_start in range(0, len(pending), batch_size):
            batch_indices = pending[batch_start : batch_start + batch_size]
            prompt_tensors = [ds[i]["question"] for i in batch_indices]
            prompts, attn, _raw_lens = self._collate_prompts(prompt_tensors)
            prompts = prompts.to(self.device)
            attn = attn.to(self.device)
            prompt_len = prompts.shape[1]

            gen_out = generate(
                self.model,
                prompts,
                attention_mask=attn,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0,
                cfg_scale=self.cfg,
                remasking=self.remasking,
                mask_id=self.mask_id,
                record_trace=record_trace,
                prompt_len=prompt_len,
                tokenizer=self.tokenizer if record_trace else None,
            )
            if record_trace:
                generated_ids, traces = gen_out
            else:
                generated_ids = gen_out
                traces = None

            for bi, i in enumerate(batch_indices):
                elem = ds[i]
                req = requests[i]
                plen = prompt_len
                stop_tokens = elem["until"]
                prompt_text = elem["question_text"]

                generated_answer = self.tokenizer.decode(
                    generated_ids[bi][plen:], skip_special_tokens=False
                )
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

                generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
                generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)

                out[i] = generated_answer
                doc_id = getattr(req, "doc_id", i)
                gold_answer = None
                doc = getattr(req, "doc", None)
                if isinstance(doc, dict):
                    gold_answer = doc.get("answer")
                trace_path = None
                if traces is not None:
                    trace_path = self._save_trace(
                        i,
                        doc_id,
                        prompt_text,
                        traces[bi],
                        generated_answer,
                        gold_answer=gold_answer,
                    )
                self._append_checkpoint(
                    i,
                    doc_id,
                    generated_answer,
                    prompt_text=prompt_text,
                    trace_path=trace_path,
                )
                pbar.update(1)

        pbar.close()
        self._flush_trace_writes()

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
