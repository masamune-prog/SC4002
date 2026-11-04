#!/usr/bin/env bash
#SBATCH --job-name=sc4002-ens-final
#SBATCH --partition=UGGPU-TC1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
.venv/bin/python ens/ensemble.py --final --name stage2_mixed --threshold 0.6 --members \
  abl_nolex_s42 abl_nolex_s43 nl_glu_span_s42 nl_glu_span_s43 nl_lstm_crf_s42 nl_tf_crf_s42
