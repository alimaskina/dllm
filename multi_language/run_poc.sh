#!/usr/bin/env bash
# Minimal PoC pipeline for multilingual diffusion feasibility checks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/multi_language"
mkdir -p results

echo "=== Step 1: Tokenization audit (CPU) ==="
python3 tokenization_audit.py --tokenizer all_unique --max-samples 5000 \
  --out results/tokenization_audit.json

echo ""
echo "=== Step 2: Training objective audit (CPU) ==="
for tok in llada qwen; do
  python3 training_objective_audit.py --tokenizer "$tok" --max-samples 5000 \
    --out "results/objective_audit_${tok}.json"
done

echo ""
echo "=== Step 3a: Oracle random trajectory sanity check (CPU) ==="
python3 oracle_trajectory_random.py --tokenizer llada --max-samples 500 \
  --out results/oracle_random_llada.json

echo ""
echo "=== Step 1b: Language competence MCQ (GPU) ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 language_competence_mcq.py \
  --model qwen15 --n-per-lang 50 --device cuda:0 \
  --out results/competence_qwen15_mcq.json

echo ""
echo "=== Step 3b: Oracle confidence trajectory with LLaDA (GPU) ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python3 oracle_trajectory_model.py \
  --model llada --max-samples 100 --device cuda:0 \
  --remasking low_confidence \
  --out results/oracle_confidence_llada.json

echo ""
echo "=== Step 3c: Free decoding trajectories (GPU) ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python3 free_decoding_trajectory.py \
  --setup both --max-samples 80 --device cuda:0 \
  --out results/free_decoding_llada.json

echo ""
echo "=== Step 4: NLL on oracle trajectories (GPU) ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 nll_oracle_audit.py \
  --max-samples 80 --device cuda:0 \
  --out results/nll_oracle_llada.json

echo ""
echo "=== Step 4b: NLL oracle + free with controls (GPU) ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 nll_decoding_audit.py \
  --max-samples 80 --device cuda:0 \
  --out results/nll_decoding_llada.json

echo ""
echo "=== Done. Results in multi_language/results/ ==="
ls -la results/
