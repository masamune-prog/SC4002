#!/usr/bin/env bash
# Three-seed hybrid masked-diffusion + CRF slot experiment.
#SBATCH --job-name=sc4002-diffusion-crf
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p checkpoints/diffusion_crf_glove logs/diffusion_crf_glove outputs/diffusion_crf_glove
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

for seed in 42 43 44; do
  echo "hybrid diffusion seed ${seed}"
  uv run python -u main_diffusion_crf_glove.py \
    --glove data/glove.6B.zip --seed "${seed}" \
    --checkpoint-dir "checkpoints/diffusion_crf_glove/seed${seed}" \
    --log-dir "logs/diffusion_crf_glove/seed${seed}"
done
