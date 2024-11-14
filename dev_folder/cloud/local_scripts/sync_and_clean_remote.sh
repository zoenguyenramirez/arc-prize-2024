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

# Function to log messages with timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

while true; do
    # Sync files from remote
    log_message "Starting sync from remote..."
    if bash cloud/local_scripts/scp_from_remote.sh barc; then
        log_message "Sync completed successfully"
    else
        log_message "Sync failed"
    fi

    # Clean up old files on remote
    log_message "Starting remote cleanup..."
    if ssh -p $PORT $REMOTE_USER@$REMOTE_HOST "cd /home/user/arc && bash cloud/remote_scripts/cleanup_old_pt_files.sh"; then
        log_message "Cleanup completed successfully"
    else
        log_message "Cleanup failed"
    fi

    # Wait for 5 minutes before next iteration
    log_message "Waiting for 5 minutes before next sync..."
    sleep 2400
done