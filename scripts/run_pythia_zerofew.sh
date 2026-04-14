#!/bin/bash

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit

# Configuration
DATA_PATH="data/processed/test.jsonl"
OUTPUT_DIR="results/pythia"
MAX_TOKENS=256
NUM_SAMPLES=3000
SAMPLING_METHOD="stratified"
SEED=42

# Arrays of models, regimes, and masking to iterate over
MODELS=("160m" "410m" "1.4b")
REGIMES=("zeroshot" "fewshot")
MASKING_FLAGS=("" "--apply_masking")

echo "Starting Pythia evaluation..."
echo "Configuration: $NUM_SAMPLES samples ($SAMPLING_METHOD), Max Tokens=$MAX_TOKENS"

for model in "${MODELS[@]}"; do
    for regime in "${REGIMES[@]}"; do
        for masking_flag in "${MASKING_FLAGS[@]}"; do
            
            # Label for display
            if [ -z "$masking_flag" ]; then
                masking_label="unmasked"
            else
                masking_label="masked"
            fi

            echo "----------------------------------------------------"
            echo "Model: Pythia-${model} | Regime: ${regime} | Masking: ${masking_label}"
            echo "----------------------------------------------------"
            
            # Adjust batch size
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
                --num_samples "$NUM_SAMPLES" \
                --sampling_method "$SAMPLING_METHOD" \
                --seed "$SEED" \
                --max_tokens "$MAX_TOKENS" \
                --match_ids_from "results/pythia/preds_pythia_1.4b_fewshot.jsonl" \
                $masking_flag
        done
    done
done

echo "Evaluation complete. Results saved to ${OUTPUT_DIR}"
