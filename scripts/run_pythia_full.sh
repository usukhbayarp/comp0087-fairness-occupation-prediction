#!/bin/bash

# ============================================================
# run_pythia_full.sh
# Full inference on the complete test set (~90k examples).
# Runs all 12 configurations: 3 models x 2 regimes x 2 masking variants.
# Redirects HuggingFace cache to /tmp to avoid home-dir quota issues.
# ============================================================

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit

# Redirect HuggingFace model cache to /tmp to avoid disk quota on home dir
export HF_HOME=/tmp/giulizhu/hf_cache
export TRANSFORMERS_CACHE=/tmp/giulizhu/hf_cache

# Configuration — no NUM_SAMPLES means full dataset
DATA_PATH="data/processed/test.jsonl"
OUTPUT_DIR="results/pythia_full"
MAX_TOKENS=256
SEED=42

MODELS=("160m" "410m" "1.4b")
REGIMES=("zeroshot" "fewshot")
MASKING_FLAGS=("" "--apply_masking")

mkdir -p "$OUTPUT_DIR"

echo "Starting full Pythia evaluation (all ~90k examples, all 12 configs)..."
echo "HF cache: $HF_HOME"
echo "Output dir: $OUTPUT_DIR"

TOTAL=0
FAILED=0

for model in "${MODELS[@]}"; do
    for regime in "${REGIMES[@]}"; do
        for masking_flag in "${MASKING_FLAGS[@]}"; do

            # Derive a readable label for this config
            if [ -z "$masking_flag" ]; then
                masking_label="unmasked"
            else
                masking_label="masked"
            fi

            echo "----------------------------------------------------"
            echo "Model: Pythia-${model} | Regime: ${regime} | Masking: ${masking_label}"
            echo "----------------------------------------------------"

            # Adjust batch size to prevent OOM
            BATCH_SIZE=32
            if [ "$model" == "1.4b" ]; then
                BATCH_SIZE=16
            fi

            python src/models/pythia/pythia_zerofew.py \
                --model_size "$model" \
                --regime "$regime" \
                --data_path "$DATA_PATH" \
                --output_dir "$OUTPUT_DIR" \
                --batch_size "$BATCH_SIZE" \
                --seed "$SEED" \
                --max_tokens "$MAX_TOKENS" \
                $masking_flag

            EXIT_CODE=$?
            TOTAL=$((TOTAL + 1))
            if [ $EXIT_CODE -ne 0 ]; then
                echo "WARNING: Run failed for pythia-${model} ${regime} ${masking_label} (exit code $EXIT_CODE)"
                FAILED=$((FAILED + 1))
            fi

        done
    done
done

echo "===================================================="
echo "Evaluation complete. ${TOTAL} runs attempted, ${FAILED} failed."
echo "Results saved to ${OUTPUT_DIR}"
echo "===================================================="
