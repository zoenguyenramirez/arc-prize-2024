#!/bin/bash

# Set the size threshold in bytes (e.g., 1MB = 1048576 bytes)
SIZE_THRESHOLD=48576

find ./runs/ -type d -links 2 | while read -r dir; do
    # Calculate total size of files starting with "events.out.tfevents." in the current directory
    total_size=$(find "$dir" -maxdepth 1 -type f -name "events.out.tfevents.*" -print0 | du -cb --files0-from=- | tail -n1 | cut -f1)

    # If total_size is empty (no matching files), set it to 0
    total_size=${total_size:-0}

    # Check if the total size is less than the threshold
    if [ "$total_size" -lt "$SIZE_THRESHOLD" ]; then
        echo "Deleting directory: $dir (size: $total_size bytes)"
        rm -rf "$dir"
    else
        echo "Keeping directory: $dir (size: $total_size bytes)"
    fi
done
