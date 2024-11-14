#!/bin/bash

# Set the output file
OUTPUT_FILE="system_info_report.txt"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$OUTPUT_FILE"
}

# Function to run a command and log its output
run_command() {
    local cmd="$1"
    local description="$2"
    
    log_message "Running: $description"
    if output=$(eval "$cmd" 2>&1); then
        echo "$output" >> "$OUTPUT_FILE"
        log_message "Success: $description"
    else
        log_message "Error: $description failed"
        echo "Error output: $output" >> "$OUTPUT_FILE"
    fi
    echo "" >> "$OUTPUT_FILE"  # Add a blank line for readability
}

# Start the report
log_message "Starting System Information Report"

# System Information
run_command "uname -a" "System Information"

# CPU Information
run_command "lscpu" "CPU Information"

# Memory Information
run_command "free -h" "Memory Information"

# Disk Space
run_command "df -h" "Disk Space"

# GPU Information (if NVIDIA GPU is present)
run_command "nvidia-smi" "GPU Information"

# Python Version
run_command "python --version" "Python Version"

# Installed Python Packages
run_command "pip list" "Installed Python Packages"

# CUDA Version
run_command "nvcc --version" "CUDA Version"

# Network Interfaces
run_command "ip addr" "Network Interfaces"

# Open Ports
run_command "ss -tuln" "Open Ports"

# Current User and Groups
run_command "id" "Current User and Groups"

# End the report
log_message "System Information Report Completed"

echo "Report generated in $OUTPUT_FILE"
