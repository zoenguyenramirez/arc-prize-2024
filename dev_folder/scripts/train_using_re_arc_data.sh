#!/bin/bash

# Function to prepare data
prepare_data() {
    echo "Preparing data..."
    python -m src.prepare_data --data-array /home/nikola/Code/GenII/re-arc/44_5000
}

# Function to train the model
train_model() {
    echo "Training model..."
    python -m src.train --epochs 500 --max-seq-length 2048 --embed-size 768 --num-layers 7 --batch-size 8 --samples-per-epoch 240 --accumulation-steps 10 --learning-rate 2e-4
}

# Function to print usage
print_usage() {
    echo "Usage: $0 [--no-prepare] [--no-train]"
    echo "  --no-prepare: Skip data preparation"
    echo "  --no-train: Skip model training"
}

# Main execution
main() {
    local skip_prepare=false
    local skip_train=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-prepare)
                skip_prepare=true
                shift
                ;;
            --no-train)
                skip_train=true
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # Source the GPU memory check script
    source "$(dirname "$0")/utils/check_gpu_memory.sh"

    if [ "$skip_prepare" = false ]; then
        prepare_data
    else
        echo "Skipping data preparation as --no-prepare argument was provided."
    fi

    if [ "$skip_train" = false ]; then
        train_model
    else
        echo "Skipping training as --no-train argument was provided."
    fi
}

# Run the main function with command line arguments
main "$@"