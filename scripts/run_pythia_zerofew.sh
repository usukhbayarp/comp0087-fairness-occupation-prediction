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

# Arrays of models and regimes to iterate over
MODELS=("160m" "410m" "1.4b")
REGIMES=("zeroshot" "fewshot")

echo "Starting Pythia evaluation..."
echo "Configuration: $NUM_SAMPLES samples ($SAMPLING_METHOD), Max Tokens=$MAX_TOKENS"

for model in "${MODELS[@]}"; do
    for regime in "${REGIMES[@]}"; do
        echo "----------------------------------------------------"
        echo "Evaluating Model: Pythia-${model} | Regime: ${regime}"
        echo "----------------------------------------------------"
        
        # We adjust batch size slightly depending on model size to prevent OOM
        # 32 for smaller models, 16 for 1.4b
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
            --max_tokens "$MAX_TOKENS"
            
    done
done

echo "Evaluation complete. Results saved to ${OUTPUT_DIR}"
