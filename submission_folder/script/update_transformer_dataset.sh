#!/bin/bash

kaggle_user=zoenguyenramirez

set -e  # Exit on error

# Check if folder name is provided as second argument
if [ -z "$2" ]; then
    echo "Usage: $0 [upload|compare|create] <folder_name>"
    exit 1
fi

folder_name="$2"

# Check if folder directory exists
if [ ! -d "$folder_name" ]; then
    mkdir -p "$folder_name"
fi

# if user wants to upload
if [ "$1" = "upload" ]; then
    # Check if source directory exists
    if [ ! -d "../input/$folder_name" ]; then
        echo "Error: ../input/$folder_name directory not found"
        exit 1
    fi

    rm -rf "$folder_name"/*
    rsync -av --exclude "__pycache__" --exclude ".gitignore" "../input/$folder_name/" "$folder_name/"
    git_hash=$(git rev-parse --short HEAD)
    combined_md5=$(find "$folder_name" -type f -exec md5sum {} \; | sort | md5sum | awk '{print $1}')
    # Get current date and time
    version=$(date '+%Y-%m-%d_%H-%M-%S')_${combined_md5}_${git_hash}
    echo "Version:" $version
    echo "$version" > "$folder_name/__version__.txt"
    (cd "$folder_name" && kaggle datasets metadata $kaggle_user/$folder_name) || exit 1

    # Update kaggle dataset with current date in the message
    kaggle datasets version --dir-mode zip -p "$folder_name" -m "$version"
    echo "Dataset uploaded successfully"
elif [ "$1" = "compare" ]; then
    kaggle datasets download $kaggle_user/$folder_name

    # Create a temporary directory for the downloaded files
    temp_dir="${folder_name}_temp"
    mkdir -p "$temp_dir"
    
    # Unzip the downloaded file to the temporary directory
    unzip -q -o "$folder_name.zip" -d "$temp_dir"
    
    # Use meld to compare directories
    meld "$temp_dir" "../input/$folder_name/"
    
    # Clean up
    rm -rf "$temp_dir" "$folder_name.zip"

elif [ "$1" = "create" ]; then
    (cd "$folder_name" && kaggle datasets metadata $kaggle_user/$folder_name) || exit 1
else
    echo "Usage: $0 [upload|compare|create] <folder_name>"
fi