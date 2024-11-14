#!/bin/bash

# Get the full path of the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the selected .env file using bash to execute select_env.sh
SELECTED_ENV_FILE=$(bash "$SCRIPT_DIR/select_env.sh")
if [ $? -ne 0 ]; then
    echo "$SELECTED_ENV_FILE"
    exit 1
fi

# Load environment variables from the selected .env file
source "$SELECTED_ENV_FILE"

# Configuration
REMOTE_FOLDER="/home/user/arc"

# Check if REMOTE_HOST is set
if [ -z "$REMOTE_HOST" ]; then
    log_message "Error: REMOTE_HOST is not set in the .env file"
    exit 1
fi

# Check if PORT is set
if [ -z "$PORT" ]; then
    log_message "Error: PORT is not set in the .env file"
    exit 1
fi

# Check if LOCAL_FOLDER is set
if [ -z "$LOCAL_FOLDER" ]; then
    log_message "Error: LOCAL_FOLDER is not set in the .env file"
    exit 1
fi

# List of subfolders to copy (add your subfolders here)
SUBFOLDERS=(
    "runs"
    "temp"
)

# List of individual files to copy
FILES_TO_COPY=()

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to run a command and handle errors
run_command() {
    local cmd="$1"
    local description="$2"
    
    log_message "Running: $description"
    if eval "$cmd"; then
        log_message "Success: $description"
    else
        log_message "Error: $description failed"
        log_message "Error output: See above"
        return 1
    fi
}

# Main script
log_message "Starting transfer from remote to local"

# Check if a custom folder name is provided as an argument
if [ $# -eq 1 ]; then
    LOCAL_FOLDER_WITH_TIMESTAMP="$LOCAL_FOLDER/cloud_runs/$REMOTE_HOST/$1"
else
    # Create a timestamp
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    # Create the local folder path with timestamp
    LOCAL_FOLDER_WITH_TIMESTAMP="$LOCAL_FOLDER/cloud_runs/$REMOTE_HOST/${TIMESTAMP}"
fi

# Create the directory
mkdir -p "$LOCAL_FOLDER_WITH_TIMESTAMP"

# Prepare rsync options
RSYNC_OPTS="-avz --progress \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.vscode' \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='*.swp' \
    -e 'ssh -p $PORT'"

# Copy each subfolder and file from remote to local
for item in "${SUBFOLDERS[@]}" "${FILES_TO_COPY[@]}"; do
    RSYNC_CMD="rsync $RSYNC_OPTS $REMOTE_USER@$REMOTE_HOST:$REMOTE_FOLDER/$item $LOCAL_FOLDER_WITH_TIMESTAMP"
    run_command "$RSYNC_CMD" "Copying $item from remote server to local"
done

log_message "Transfer completed"
