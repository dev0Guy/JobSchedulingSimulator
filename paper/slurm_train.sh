#!/bin/bash
# Train + evaluate the pointer network on a SLURM GPU node.
#   sbatch paper/slurm_train.sh          (from the repo root)
#   squeue --me / scancel <jobid> / tail -f paper/out/slurm-<jobid>.out
# Override anything from the command line, e.g.
#   sbatch --export=ALL,ITERS=100,SIMS=128 paper/slurm_train.sh

#SBATCH --partition main
#SBATCH --time 6-23:59:59
#SBATCH --job-name ptr-mcts
#SBATCH --output paper/out/slurm-%J.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=17            ### 16 self-play workers + 1 main process
#SBATCH --mem=40G
##SBATCH --mail-user=YOUR_EMAIL
##SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
echo "$(date)  job $SLURM_JOBID on $SLURM_JOB_NODELIST, $SLURM_CPUS_PER_TASK cpus"

REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$REPO/.venv/bin/python}       # python with torch + numpy + gymnasium
cd "$REPO"
mkdir -p paper/out

ITERS=${ITERS:-50}
SIMS=${SIMS:-64}
WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))
EPISODES=${EPISODES:-$(( WORKERS * 2 ))}   # a multiple of the worker count
SEED=${SEED:-0}
OUT=${OUT:-paper/out/ptr_s${SEED}.pt}

zig build lib -Doptimize=ReleaseFast

srun "$PY" -u paper/train.py --device cuda --workers "$WORKERS" --seed "$SEED" \
    --iters "$ITERS" --episodes "$EPISODES" --sims "$SIMS" \
    --batch 512 --updates 200 --out "$OUT"

srun "$PY" -u paper/evaluate.py "$OUT" --episodes 20 --sims "$SIMS" --device cuda \
    --out "${OUT%.pt}_eval.json"
echo "$(date)  done"
