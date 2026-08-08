#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/home/alimaskina/.cache/huggingface

cd /home/alimaskina/dllm/quant_unmask

conda run -n llada_quant python exp1_fidelity.py \
  --model GSAI-ML/LLaDA-8B-Base \
  --fp-quant fp16 \
  --q-quant int4 \
  --n-samples 5 \
  --comp-len 16 \
  --out results_exp1_smoke.json \
  2>&1 | tee exp1_smoke.log
