#!/usr/bin/env bash
# Submit from the repository root: sbatch train_transformer_crf.sbatch
#SBATCH --job-name=sc4002-transformer-crf-deep
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
mkdir -p checkpoints/transformer_crf_deep logs/transformer_crf_deep outputs/transformer_crf_deep
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

uv run python -u main_transformer_crf.py \
  --train-csv data/train.csv \
  --dev-csv data/dev.csv \
  --test-csv data/test.csv \
  --tokenizer data/bpe_tokenizer.json \
  --checkpoint-dir checkpoints/transformer_crf_deep \
  --log-dir logs/transformer_crf_deep

uv run python -u eval.py \
  --checkpoint checkpoints/transformer_crf_deep/best_transformer_crf.pt \
  --test-csv data/test.csv \
  --output-dir outputs/transformer_crf_deep
