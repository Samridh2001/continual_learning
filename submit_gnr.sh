#!/bin/bash
#SBATCH --job-name=dyn_expand
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

echo "========================================================================"
echo "Job: Dynamic Zero-Data Direct-Sum Dimension Expansion"
echo "Started on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID | User: $(whoami) | Node: $SLURMD_NODENAME"
echo "Method: Tasks 0-4 run at native 768-D with Pure Augmentation Purging."
echo "        At Task 5: Zero-data direct-sum expansion (+1280 dims -> 2048-D)."
echo "        Past heads 0-4 are zero-padded: W_new = [W_old, 0] (0 past images needed!)."
echo "        Tasks 5-9 train in expanded 2048-D space with fresh capacity."
echo "Config: configs/mimg10_dynamic_expansion.json"
echo "========================================================================"

set -e

cd /users/student/pg/pg25/samridh/project_gnr/InfLoRA
export PYTHONPATH=".:$PYTHONPATH"

mkdir -p logs
LOG_FILE="logs/dyn_expand_${SLURM_JOB_ID}_$(date +'%Y%m%d_%H%M%S').log"
echo "Saving execution log to: $LOG_FILE"

/users/student/pg/pg25/samridh/miniconda3/envs/inflora/bin/python -u main.py \
    --device 0 \
    --config configs/mimg10_dynamic_expansion.json 2>&1 | tee "$LOG_FILE"

echo "========================================================================"
echo "Finished at $(date)"
echo "========================================================================"
