#!/bin/bash

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

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    log_message "Found requirements.txt. Installing packages..."
    run_command "pip install -r requirements.txt" "Installing packages from requirements.txt"
else
    log_message "Error: requirements.txt not found in the current directory."
    exit 1
fi

# Install nvitop
run_command "pip install nvitop" "Installing nvitop"

# Final message
log_message "Installation complete. Please restart your terminal or run 'source ~/.bashrc' to use the new environment."