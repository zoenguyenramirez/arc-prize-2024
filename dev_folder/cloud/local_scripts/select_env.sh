#!/bin/bash

# Name of the environment variable that will determine which .env file to use
ENV_SELECTOR=${ENV_SELECTOR:-"default"}

# Base directory for .env files
ENV_DIR="$(dirname "$0")/env_files"

# Construct the path to the selected .env file
SELECTED_ENV_FILE="${ENV_DIR}/.env.${ENV_SELECTOR}"

# Check if the selected .env file exists
if [ -f "$SELECTED_ENV_FILE" ]; then
    echo "$SELECTED_ENV_FILE"
else
    echo "Error: .env file for $ENV_SELECTOR not found at $SELECTED_ENV_FILE" >&2
    exit 1
fi