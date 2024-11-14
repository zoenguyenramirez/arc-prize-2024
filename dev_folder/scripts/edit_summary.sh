#!/bin/bash

# Check if a path argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide the path to your TensorBoard log directory as an argument."
    echo "Usage: $0 /path/to/your/tensorboard/log/directory"
    exit 1
fi

# Set the log directory from the first command-line argument
TENSORBOARD_LOG_DIR="$1"

# Check if the provided directory exists
if [ ! -d "$TENSORBOARD_LOG_DIR" ]; then
    echo "Error: The specified directory does not exist: $TENSORBOARD_LOG_DIR"
    exit 1
fi

# Run the Python script to add custom summary
python -m scripts.edit_summary "$TENSORBOARD_LOG_DIR"

echo "Custom summary added successfully to $TENSORBOARD_LOG_DIR."
