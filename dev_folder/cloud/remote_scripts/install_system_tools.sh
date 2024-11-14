#!/bin/bash

# Function to log messages (copy from the original script)
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to run a command and handle errors (copy from the original script)
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

# Install tmux
run_command "sudo apt-get update && sudo apt-get install -y mc" "Installing mc"

log_message "System tools installation complete."