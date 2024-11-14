#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <directory> <file_pattern>"
    echo "Finds the latest file matching the specified pattern in the given directory"
    echo "Example: $0 /path/to/directory '*.pth'"
    exit 1
}

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    usage
fi

# Get the directory and file pattern from arguments
directory="$1"
file_pattern="$2"

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "Error: Directory '$directory' does not exist."
    exit 1
fi

# Find the latest file matching the pattern
latest_file=$(find "$directory" -type f -name "$file_pattern" -printf '%T@ %p\n' | 
              sort -n | 
              tail -n 1 | 
              cut -f2- -d" ")

# Check if a file was found
if [ -z "$latest_file" ]; then
    echo "No files matching '$file_pattern' found in '$directory'"
else
    echo "Latest file: $latest_file"
fi
