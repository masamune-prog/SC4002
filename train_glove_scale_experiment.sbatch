#!/usr/bin/env bash
# Compare 3.05M and 6.94M GloVe CRF models with three seeds each.
# The baseline seed 42 run is already saved under checkpoints/glove_crf/.
#SBATCH --job-name=sc4002-glove-scale
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/glove_scale checkpoints/glove_scale outputs/glove_scale
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

for seed in 43 44; do
  echo "baseline seed ${seed}"
  uv run python -u main_glove_crf.py \
    --glove data/glove.6B.zip --seed "${seed}" \
    --checkpoint-dir "checkpoints/glove_scale/baseline_seed${seed}" \
    --log-dir "logs/glove_scale/baseline_seed${seed}"
done

for seed in 42 43 44; do
  echo "scaled seed ${seed}"
  uv run python -u main_glove_crf.py \
    --glove data/glove.6B.zip --seed "${seed}" \
    --d-model 384 --nhead 6 --num-layers 4 --dim-feedforward 768 \
    --checkpoint-dir "checkpoints/glove_scale/scaled_seed${seed}" \
    --log-dir "logs/glove_scale/scaled_seed${seed}"
done
