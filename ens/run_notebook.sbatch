#!/usr/bin/env bash
# Execute main.ipynb on a GPU node, saving all outputs into it.  Usage: sbatch ens/run_notebook.sbatch [extra args]
#SBATCH --job-name=sc4002-notebook
#SBATCH --partition=UGGPU-TC1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python ens/run_notebook.py main.ipynb "$@"
