#!/bin/bash

# Configuration
NSYS_PATH="$HOME/nsight-systems-2024.6.1/bin/nsys"
NSYS_UI_PATH="$HOME/nsight-systems-2024.6.1/bin/nsys-ui"
PYTHON_SCRIPT="src.train"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="nsight_profile_report_${TIMESTAMP}"
EPOCHS=3

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if nsys is installed
if ! command_exists "$NSYS_PATH"; then
    echo "Error: nsys is not installed or not in the specified path."
    exit 1
fi

# Run nsys profile
echo "Starting Nsight Systems profiling..."
"$NSYS_PATH" profile \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -o "$OUTPUT_FILE" \
    python -m "$PYTHON_SCRIPT" --epochs "$EPOCHS"

# Check if profiling was successful
if [ $? -eq 0 ]; then
    echo "Profiling completed successfully."
    echo "To view the results, run:"
    echo "$NSYS_UI_PATH"
else
    echo "Error: Profiling failed."
    exit 1
fi

# Uncomment the following line to automatically open the UI after profiling
"$NSYS_UI_PATH" "$OUTPUT_FILE.nsys-rep" &

exit 0