#!/usr/bin/env bash
# Run every line of a command file, N at a time, on one GPU.  Usage: sbatch ens/run_queue.sbatch <cmdfile> [N]
#SBATCH --job-name=sc4002-ens
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
nvidia-smi --query-gpu=name,memory.total --format=csv
N="${2:-4}"
xargs -P "$N" -I{} bash -c '{}' < "$1"
echo "QUEUE DONE"
