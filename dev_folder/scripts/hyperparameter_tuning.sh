#!/bin/bash

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
# Source the process manager script
source "$scriptDir/process_manager.sh"
source "$scriptDir/utils/suppress_warnings.sh"

# Function to display usage
show_usage() {
    echo "Usage: $0 [--no-process-manager] [--runs-name <name>] [--load-checkpoint <checkpoint>] [--scenario <scenario>]"
    echo "  --no-process-manager: Launch the job directly without using the process manager"
    echo "  --runs-name <name>: Specify a name for the run"
    echo "  --load-checkpoint <checkpoint>: Specify a checkpoint to load"
    echo "  --scenario <scenario>: Specify the hyperparameter scenario to use"
}

# Set default values
USE_PROCESS_MANAGER=true
RUNS_NAME=""
LOAD_CHECKPOINT=""
SCENARIO=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-process-manager)
            USE_PROCESS_MANAGER=false
            shift
            ;;
        --runs-name)
            RUNS_NAME="$2"
            shift 2
            ;;
        --load-checkpoint)
            LOAD_CHECKPOINT="$2"
            shift 2
            ;;
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Construct the command with the runs_name, load_checkpoint, and scenario variables
COMMAND="python -m src.hyperparameter_tuning"
if [ -n "$RUNS_NAME" ]; then
    COMMAND+=" --runs-name \"$RUNS_NAME\""
fi
if [ -n "$LOAD_CHECKPOINT" ]; then
    COMMAND+=" --load-checkpoint \"$LOAD_CHECKPOINT\""
fi
if [ -n "$SCENARIO" ]; then
    COMMAND+=" --scenario \"$SCENARIO\""
fi

# Source the GPU memory check script
source "$(dirname "$0")/utils/check_gpu_memory.sh"

if $USE_PROCESS_MANAGER; then
    if is_process_running ; then
        echo "A hyperparameter tuning process is already running."
        echo "To kill it, run: ./kill_process.sh"
    else
        start_process "$COMMAND"
    fi
else
    echo "Launching job directly without process manager..."
    eval "$COMMAND"
fi
