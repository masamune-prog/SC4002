#!/usr/bin/env bash
# Long run of the original 6.94M GloVe Transformer + CRF, seed 43.
#SBATCH --job-name=sc4002-glove-scale-long
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p checkpoints/glove_scale_long/seed43 logs/glove_scale_long/seed43
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

uv run python -u main_glove_crf.py \
  --glove data/glove.6B.zip --glove-dim 100 \
  --seed 43 \
  --d-model 384 --nhead 6 --num-layers 4 --dim-feedforward 768 \
  --epochs 1000 --patience 1000 \
  --checkpoint-dir checkpoints/glove_scale_long/seed43 \
  --log-dir logs/glove_scale_long/seed43
