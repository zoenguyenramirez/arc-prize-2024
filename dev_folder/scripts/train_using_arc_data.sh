#!/bin/bash

# Function to prepare data
prepare_data() {
    echo "Preparing data..."
    python -m src.prepare_data --data-sources arc-agi_training arc-agi_evaluation
}

# Function to train the model
train_model() {
    echo "Training model..."
    python -m src.train --epochs 500 --dataset-file intermediate_data/prepared_dataset_using_arc.pth --max-seq-length 2048 --embed-size 48 --batch-size 22
}

# Main execution
main() {
    prepare_data
    if [ "$1" != "--no-train" ]; then
        train_model
    else
        echo "Skipping training as --no-train argument was provided."
    fi
}

# Run the main function with command line arguments
main "$@"
