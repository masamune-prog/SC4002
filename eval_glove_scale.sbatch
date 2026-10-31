#!/usr/bin/env bash
# Evaluate the scaled GloVe model selected by dev slot F1 (seed 43).
#SBATCH --job-name=sc4002-glove-scale-eval
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/glove_scale/scaled_seed43
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

uv run python -u eval_glove.py \
  --checkpoint checkpoints/glove_scale/scaled_seed43/best_glove_crf.pt \
  --glove data/glove.6B.zip \
  --test-csv data/test.csv \
  --output-dir outputs/glove_scale/scaled_seed43
