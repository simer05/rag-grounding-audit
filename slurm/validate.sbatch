#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --job-name=ra_val
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Validation run: one model, 50 questions, all five conditions.
# Usage (from the repository root, with your Python environment active):
#   MODEL_DIR=<local Llama-3.1-8B-Instruct folder> sbatch slurm/validate.sbatch
set -e
: "${MODEL_DIR:?set MODEL_DIR to a local model folder}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-.}"
python -m rag_audit.selftest
python -m rag_audit.cli \
    --data data/hotpot_distractor_validation.parquet \
    --n 50 --seed 13 \
    --model-path "$MODEL_DIR" \
    --model-name llama-3.1-8b \
    --out-dir results/validation_llama31_n50 \
    --batch-size 8
