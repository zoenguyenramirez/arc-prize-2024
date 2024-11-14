#!/bin/bash

# File: process_manager.sh

# Function to start a process and save its PID
start_process() {
    local command="$1"
    local pgid_file=".TASK_PGID"

    # Start the process in the background
    eval "$command" &
    local pid=$!
    local pgid=$(ps -o pgid= -p $pid)

    # Save the PID to a file
    echo $pgid > "$pgid_file"

    echo "Process started with PID $pid. To kill it, run: scripts/kill_training_process.sh"
}

# Function to check if a process is running
is_process_running() {
    local pgid_file=".TASK_PGID"
    if [ -f "$pgid_file" ]; then
        local pid=$(cat "$pgid_file")
        if kill -0 $pid 2>/dev/null; then
            return 0  # Process is running
        else
            return 1  # Process is not running
        fi
    else
        return 1  # PID file doesn't exist
    fi
}