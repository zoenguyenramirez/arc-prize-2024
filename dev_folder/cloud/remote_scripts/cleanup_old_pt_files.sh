#!/bin/bash

# Default values
target_dir="runs"
keep_count=3

# Function to display usage information
usage() {
    echo "Usage: $0 [-d directory] [-k keep_count]"
    echo "  -d  Target directory (default: current directory)"
    echo "  -k  Number of newest files to keep (default: 3)"
    exit 1
}

# Parse command-line arguments
while getopts ":d:k:" opt; do
    case $opt in
        d) target_dir="$OPTARG" ;;
        k) keep_count="$OPTARG" ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

# Validate arguments
if [ ! -d "$target_dir" ]; then
    echo "Error: Directory '$target_dir' does not exist." >&2
    exit 1
fi

if ! [[ "$keep_count" =~ ^[0-9]+$ ]]; then
    echo "Error: Keep count must be a positive integer." >&2
    exit 1
fi

# Find all *.pt files, sort by modification time, and store in an array
mapfile -t files < <(find "$target_dir" -type f -name "*.pt" -print0 | xargs -0 ls -t)

# Calculate the number of files to delete
delete_count=$((${#files[@]} - keep_count))

# Delete excess files
if [ $delete_count -gt 0 ]; then
    for ((i=keep_count; i<${#files[@]}; i++)); do
        rm "${files[i]}"
        echo "Deleted: ${files[i]}"
    done
    echo "Deleted $delete_count file(s). Kept the newest $keep_count file(s)."
else
    echo "Found ${#files[@]} file(s). No files deleted as the count is less than or equal to $keep_count."
fi
