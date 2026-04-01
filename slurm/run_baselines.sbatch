#!/bin/bash
#SBATCH --job-name=steer-baseline
#SBATCH --partition=h200x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/nvwulf/scratch/nijjohnson/logs/baseline_%j.out
#SBATCH --error=/lustre/nvwulf/scratch/nijjohnson/logs/baseline_%j.err

# --- Environment Setup ---
module load cuda12.8/toolkit/12.8.1
conda activate steering-study

export HF_HOME=/lustre/nvwulf/scratch/nijjohnson/hf_cache
export HF_TOKEN=$(cat ${HOME}/.cache/huggingface/token)

# --- Working Directory ---
cd /lustre/nvwulf/scratch/nijjohnson/activation-steering-study

# --- Run Baselines ---
python scripts/run_baselines.py \
    --config configs/default.yaml \
    ${MODEL:+--model $MODEL} \
    ${PERSONA:+--persona $PERSONA}

echo "Baseline job $SLURM_JOB_ID completed at $(date)"
