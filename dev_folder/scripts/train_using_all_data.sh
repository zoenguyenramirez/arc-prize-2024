#!/bin/bash

# Function to prepare data
prepare_data() {
    echo "Preparing data..."
    python -m src.prepare_data --data-sources synth_conditional_logic synth_array_indexing synth_arithmetic_operations synth_modular_arithmetic synth_find_min_max synth_count_occurrences synth_element_wise_operations synth_array_manipulation pattern arc-agi_training arc-agi_evaluation
}

# Function to train the model
train_model() {
    echo "Training model..."
    python -m src.train --epochs 5000
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