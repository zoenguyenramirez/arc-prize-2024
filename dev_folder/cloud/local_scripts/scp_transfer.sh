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
    "cloud"
    "scripts"
    "src"
    "tests"
    "input_data"
)

# List of individual files to copy (now supports full path wildcards)
FILES_TO_COPY=(
    "requirements.txt"
    "intermediate_data/prepared_dataset_re_arc_100_5000.pth"
    # "intermediate_data/prepared_dataset_re_arc_*_5000.pth.tar.xz"
    "intermediate_data/prepared_dataset_using_arc.pth"
    "intermediate_data/prepared_dataset_using_barc.pth"
    "intermediate_data/prepared_dataset_using_arc_training.pth"
)

# List of folders to create on remote machine but not copy
REMOTE_FOLDERS_TO_CREATE=(
    "intermediate_data"
    "temp"
    # Add other folders here as needed
)

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

# Function to create remote folders
create_remote_folders() {
    local folders=("$@")
    for folder in "${folders[@]}"; do
        local ssh_cmd="ssh -p $PORT $REMOTE_USER@$REMOTE_HOST 'mkdir -p $REMOTE_FOLDER/$folder'"
        run_command "$ssh_cmd" "Creating remote folder: $folder"
    done
}

# Check if REMOTE_HOST is set
if [ -z "$REMOTE_HOST" ]; then
    log_message "Error: REMOTE_HOST environment variable is not set"
    exit 1
fi

# Create remote folders
create_remote_folders "${REMOTE_FOLDERS_TO_CREATE[@]}"

# Main script
log_message "Starting transfer to remote"

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

# Copy individual files (respecting relative and absolute paths)
for file_pattern in "${FILES_TO_COPY[@]}"; do
    # If the file pattern starts with a slash, it's an absolute path
    if [[ $file_pattern == /* ]]; then
        source_path="$file_pattern"
        # Extract the relative path from LOCAL_FOLDER
        relative_path="${source_path#$LOCAL_FOLDER/}"
        dest_dir="$REMOTE_FOLDER/$(dirname "$relative_path")"
    else
        source_path="$LOCAL_FOLDER/$file_pattern"
        dest_dir="$REMOTE_FOLDER/$(dirname "$file_pattern")"
    fi
    
    RSYNC_CMD="rsync $RSYNC_OPTS $source_path $REMOTE_USER@$REMOTE_HOST:$dest_dir/"
    run_command "$RSYNC_CMD" "Copying files matching $file_pattern to remote server"
done

# Copy each subfolder and file
for item in "${SUBFOLDERS[@]}"; do
    RSYNC_CMD="rsync $RSYNC_OPTS $LOCAL_FOLDER/$item $REMOTE_USER@$REMOTE_HOST:$REMOTE_FOLDER/"
    run_command "$RSYNC_CMD" "Copying subfolder $item to remote server"
done

log_message "Transfer completed"
