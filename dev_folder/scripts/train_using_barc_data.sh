#!/bin/bash

source "$(dirname "$0")/utils/check_gpu_memory.sh"
source "$(dirname "$0")/utils/suppress_warnings.sh"

# Function to prepare data
prepare_data() {
    echo "Preparing data..."
    python -m src.prepare_data --jsonl-file /home/nikola/Code/GenII/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems/data_100k.jsonl 
}

# Function to train the model
train_model() {
    RUN_NAME="barc"

    latest_pth=$(find "runs/$RUN_NAME" -name "*.pt" -type f -printf '%T+ %p\n' | sort -r | head -n 1 | cut -d' ' -f2-)
    echo "Latest best checkpoint: $latest_pth"

    echo "Training model..."
    python -m src.train --dataset-file intermediate_data/prepared_dataset_using_arc.pth intermediate_data/prepared_dataset_using_barc.pth intermediate_data/prepared_dataset_re_arc_100_5000.pth \
            --epochs 5000 \
            --max-seq-length 2480 \
            --samples-per-epoch 800 \
            --runs-name $RUN_NAME \
            --seed 0 \
            --learning-rate 1e-4 \
            --warmup-epochs 10 \
            --heads 4 \
            --base-lr 1e-8 \
            --num-layers 18 \
            --batch-size 2 \
            --schedular cosine \
            --embed-size 888 \
            --accumulation-steps 12 \
            --progressive-head 1 \
            --mask-hack False \
            --load-checkpoint "$latest_pth"
}

# Main execution
main() {
    # prepare_data
    if [ "$1" != "--no-train" ]; then
        train_model
    else
        echo "Skipping training as --no-train argument was provided."
    fi
}

# Run the main function with command line arguments
main "$@"
