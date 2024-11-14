#!/bin/bash

# Check if a folder path is provided
if [ $# -eq 0 ]; then
    echo "Please provide a folder path"
    exit 1
fi

source "$(dirname "$0")/utils/check_gpu_memory.sh"
source "$(dirname "$0")/utils/suppress_warnings.sh"

folder_path="$1"

# Find all *.pt files, sort them by folder path, and process them
find "$folder_path" -type f -name "*.pt" | sort -r | while read -r checkpoint_path; do
    echo Processing $checkpoint_path
    
    logger_file="report/$checkpoint_path.log" # preserve the original relative path

    if [ -f "$logger_file" ]; then
        echo "Log file already exists, skipping: $logger_file"
        continue
    fi

    # Create the report directory if it doesn't exist
    mkdir -p "$(dirname "$logger_file")"

    # Run the first command
    python -m src.validate_model --checkpoint-path "$checkpoint_path"  --data-source arc-agi_training --logger-file $logger_file  # --verbose --second-only 
    
    # Run the second command
    python -m src.validate_model --checkpoint-path "$checkpoint_path"  --data-source arc-agi_evaluation --logger-file $logger_file # --verbose --second-only 
done

echo "All files processed."
