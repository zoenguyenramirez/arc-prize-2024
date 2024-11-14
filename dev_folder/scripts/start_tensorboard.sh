#!/bin/bash

# Define an associative array for options and their corresponding actions
declare -A options
options["runs"]="tensorboard --logdir=runs --host 0.0.0.0 --port=6006"
options["special-checkpoints"]="tensorboard --logdir=special_checkpoints --host 0.0.0.0 --port=6007"
options["cloud-runs"]="tensorboard --logdir=cloud_runs --host 0.0.0.0  --port=6008"
options["bug-checkpoints"]="tensorboard --logdir=bug_checkpoints --host 0.0.0.0  --port=6009"

# Function to start TensorBoard for a given option
start_tensorboard() {
    local option=$1
    local command=${options[$option]}
    if [[ -n $command ]]; then
        $command
    else
        echo "Unknown option: $option"
    fi
}

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Available options:"
    for option in "${!options[@]}"; do
        echo "  --$option"
    done
}

# Parse command-line arguments
if [[ $# -eq 0 ]]; then
    show_usage
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --*)
            option=${1#--}
            start_tensorboard "$option"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# echo "To kill tensorboard:"
# echo "kill \$(ps aux | grep tensorboard | grep -v grep | awk '{print \$2}')"
