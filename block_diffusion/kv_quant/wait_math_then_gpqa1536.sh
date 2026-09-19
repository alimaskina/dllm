#!/usr/bin/env bash
# Wait for MATH500 n=100 to finish, stop old wrapper (would launch GPQA@1024), start GPQA@1536
set -euo pipefail
LOG=/home/alimaskina/dllm/block_diffusion/checkpoints/math500_kivi_n100.log
MARK="Saved → ../checkpoints/gsm8k_math500_kivi_n100/summary.json"

echo "Waiting for MATH500 to finish..."
while ! grep -qF "$MARK" "$LOG" 2>/dev/null; do
  sleep 30
done
echo "MATH500 done."

# Stop wrapper bash so it does not launch GPQA @1024
pkill -f 'bash run_crossdataset_n100.sh' 2>/dev/null || true
sleep 2

echo "Launching GPQA @1536..."
cd /home/alimaskina/dllm/block_diffusion/kv_quant
nohup bash run_gpqa_n100.sh > ../checkpoints/gpqa_diamond_kivi_n100_launch.log 2>&1 &
echo "GPQA PID=$!"
