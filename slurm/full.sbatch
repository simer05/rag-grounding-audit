#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --job-name=ra_full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Full run for one model: 500 questions, all conditions.
# Usage (from the repository root, with your Python environment active):
#   LLAMA_DIR=... GEMMA_DIR=... MISTRAL_DIR=... sbatch slurm/full.sbatch <llama|gemma|mistral> [retrieved-only]
# Each *_DIR points to a local copy of the model weights.
set -e

case "$1" in
  llama)   NAME=llama-3.1-8b;     BS=8; MODEL_DIR=$LLAMA_DIR ;;
  gemma)   NAME=gemma-2-9b;       BS=4; MODEL_DIR=$GEMMA_DIR ;;
  mistral) NAME=mistral-nemo-12b; BS=4; MODEL_DIR=$MISTRAL_DIR ;;
  *) echo "unknown model key: $1"; exit 1 ;;
esac
: "${MODEL_DIR:?set the model directory for $1}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# Optional second argument "retrieved-only" runs just step 3 (for a model whose steps 1-2 already ran).
MODE=${2:-all}

cd "${SLURM_SUBMIT_DIR:-.}"
python -m rag_audit.selftest

if [ "$MODE" = all ]; then
    # 1) 500 questions, all five given-passage conditions.
    python -m rag_audit.cli \
        --data data/hotpot_distractor_validation.parquet \
        --n 500 --seed 13 \
        --model-path "$MODEL_DIR" \
        --model-name $NAME \
        --out-dir results/full_${1}_n500 \
        --batch-size $BS

    # 2) Conflict condition over every validation question with a numeric answer (short prompts).
    python -m rag_audit.cli \
        --data data/hotpot_distractor_validation.parquet \
        --n 7405 --seed 13 --conditions conflict \
        --model-path "$MODEL_DIR" \
        --model-name $NAME \
        --out-dir results/full_${1}_conflict_all \
        --batch-size $BS
fi

# 3) End-to-end RAG: the same 500 questions, BM25 top-5 over the pooled corpus of their passages.
python -m rag_audit.cli \
    --data data/hotpot_distractor_validation.parquet \
    --n 500 --seed 13 --conditions retrieved --top-k 5 \
    --model-path "$MODEL_DIR" \
    --model-name $NAME \
    --out-dir results/full_${1}_retrieved \
    --batch-size $BS

# 4) Re-score every saved run with the current scoring code (CPU only), then refresh the comparison table.
for d in results/full_*/; do python -m rag_audit.rescore "$d" > /dev/null; done
python -m rag_audit.compare results
