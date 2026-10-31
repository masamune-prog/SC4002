#!/usr/bin/env bash
# Train three GloVe + BiGRU + CRF seeds; select using dev slot F1.
#SBATCH --job-name=sc4002-bigru-glove
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
mkdir -p checkpoints/bigru_glove_crf logs/bigru_glove_crf outputs/bigru_glove_crf
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

for seed in 42 43 44; do
  echo "BiGRU seed ${seed}"
  uv run python -u main_bigru_glove_crf.py \
    --glove data/glove.6B.zip --seed "${seed}" \
    --checkpoint-dir "checkpoints/bigru_glove_crf/seed${seed}" \
    --log-dir "logs/bigru_glove_crf/seed${seed}"
done
