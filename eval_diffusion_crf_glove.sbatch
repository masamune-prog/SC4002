#!/usr/bin/env bash
# Evaluate the best-dev diffusion + CRF checkpoint (seed 43).
#SBATCH --job-name=sc4002-diffusion-eval
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
mkdir -p outputs/diffusion_crf_glove
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

uv run python -u eval_diffusion_glove.py \
  --checkpoint checkpoints/diffusion_crf_glove/seed43/best_diffusion_glove.pt \
  --glove data/glove.6B.zip \
  --test-csv data/test.csv \
  --output-dir outputs/diffusion_crf_glove
