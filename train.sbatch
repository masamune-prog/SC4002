#!/usr/bin/env bash
# Submit from the repository root: sbatch train.sbatch
#SBATCH --job-name=sc4002-bpe-transformer_scale_up
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
mkdir -p checkpoints/bpe_transformer_scale_up logs/bpe_transformer_scale_up outputs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

uv run python -u main_scale_up.py \
  --train-csv data/train.csv \
  --dev-csv data/dev.csv \
  --test-csv data/test.csv \
  --tokenizer data/bpe_tokenizer.json \
  --output-dir checkpoints/bpe_transformer_scale_up \
  --log-dir logs/bpe_transformer_scale_up

uv run python -u eval.py \
  --checkpoint checkpoints/bpe_transformer_scale_up/best_bpe_transformer_scale_up.pt \
  --test-csv data/test.csv \
  --output-dir outputs

