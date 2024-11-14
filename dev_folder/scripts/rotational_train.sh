#!/bin/bash

# Source the GPU memory check script
source "$(dirname "$0")/utils/check_gpu_memory.sh"
source "$(dirname "$0")/utils/suppress_warnings.sh"

# Define the two patterns
PATTERN1="intermediate_data/prepared_dataset_re_arc_*_5000.pth.tar.xz"
PATTERN2="intermediate_data/prepared_dataset_using_arc.pth"

# Use find to get all files matching the patterns
FILES1=$(find . -type f \( -path "./$PATTERN1" \) | sort)
FILES2=$(find . -type f \( -path "./$PATTERN2" \) | sort)

# Check if any files were found
if [ -z "$FILES1" ] || [ -z "$FILES2" ]; then
    echo "No files found matching the patterns"
    exit 1
fi

python -m scripts.utils.delete_invalid_pytorch_files intermediate_data/

# Run the Python script with the found files
python -m src.rotational_train --dataset-files1 $FILES1 --dataset-files2 $FILES2 --runs-name "l9_h8" --initial-index 1 --initial-lr 6e-05

# hyper parameter tested
# with mask:
# @initial case: 2e-4 (1e-4 as base lr), tested with 7 layers
# @continue case: 3e-5 (2e-6 as base lr), after iteration 7 (14 layers)
