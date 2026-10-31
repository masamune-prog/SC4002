#!/usr/bin/env bash
# Submit from repository root: sbatch train_glove_crf.sbatch
# Place Stanford glove.6B.zip in data/ before submission.
#SBATCH --job-name=sc4002-glove-crf
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
mkdir -p checkpoints/glove_crf logs/glove_crf outputs/glove_crf
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

uv run python -u main_glove_crf.py \
  --glove data/glove.6B.zip \
  --glove-dim 100 \
  --train-csv data/train.csv \
  --dev-csv data/dev.csv \
  --checkpoint-dir checkpoints/glove_crf \
  --log-dir logs/glove_crf

uv run python -u eval_glove.py \
  --checkpoint checkpoints/glove_crf/best_glove_crf.pt \
  --glove data/glove.6B.zip \
  --test-csv data/test.csv \
  --output-dir outputs/glove_crf
