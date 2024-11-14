#!/bin/bash

# Sanity Check Script for Model Training and Evaluation
# Usage: ./sanity_check.sh <batch_size>

set -e

# Check if batch size argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Batch size argument is required."
    echo "Usage: $0 <batch_size> (75 can converge to almost 0 in 20 epochs)"
    exit 1
fi

BATCH_SIZE=$1
RUN_NAME="sanity_check"
DATASET_FILE="intermediate_data/prepared_dataset.pth"

echo "Starting sanity check with batch size: $BATCH_SIZE"

# Source the GPU memory check script
source "$(dirname "$0")/utils/check_gpu_memory.sh"
source "$(dirname "$0")/utils/suppress_warnings.sh"

# Clean up previous run
rm -rf "runs/$RUN_NAME"

# Prepare data
python -m src.prepare_data --data-sources synth_conditional_logic 

# Initial training
python -m src.train \
    --epochs 10 \
    --max-seq-length 99 \
    --samples-per-epoch 10240 \
    --learning-rate 1e-2 \
    --base-lr 5e-3 \
    --step-size-up 0 \
    --batch-size "$BATCH_SIZE" \
    --heads 2 \
    --embed-size 40 \
    --num-layers 3 \
    --warmup-epochs 3 \
    --progressive-head 1 \
    --auto-cast True \
    --augment-data False \
    --schedular cosine \
    --accumulation-steps 2 \
    --dataset-file "$DATASET_FILE" \
    --runs-name "$RUN_NAME" \
    --mask-hack False

# Find the latest best checkpoint
latest_pth=$(find "runs/$RUN_NAME" -name "*_best.pt" -type f -printf '%T+ %p\n' | sort -r | head -n 1 | cut -d' ' -f2-)
echo "Latest best checkpoint: $latest_pth"

# Sample using the best checkpoint
python -m src.sample --checkpoint-path "$latest_pth" --data-source synth_conditional_logic_test

# Fine-tuning
python -m src.train \
    --epochs 5 \
    --max-seq-length 300 \
    --samples-per-epoch 10240 \
    --learning-rate 5e-03 \
    --base-lr 2e-03 \
    --step-size-up 0 \
    --batch-size "$BATCH_SIZE" \
    --heads 2 \
    --embed-size 40 \
    --num-layers 3 \
    --warmup-epochs 4 \
    --progressive-head 1 \
    --auto-cast True \
    --augment-data False \
    --schedular cosine \
    --accumulation-steps 2 \
    --dataset-file "$DATASET_FILE" \
    --runs-name "$RUN_NAME" \
    --load-checkpoint "$latest_pth" \
    --mask-hack False

# Find the new best checkpoint after fine-tuning
latest_pth=$(find "runs/$RUN_NAME" -name "*_best.pt" -type f -printf '%T+ %p\n' | sort -r | head -n 1 | cut -d' ' -f2-)
echo "New best checkpoint after fine-tuning: $latest_pth"

# Sample using the new best checkpoint
python -m src.sample --checkpoint-path "$latest_pth" --data-source synth_conditional_logic_test

echo "Sanity check completed successfully."
