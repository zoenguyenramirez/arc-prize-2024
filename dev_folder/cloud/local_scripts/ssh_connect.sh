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

# Check if required variables are set
if [ -z "$REMOTE_HOST" ] || [ -z "$PORT" ] || [ -z "$LOCAL_FOLDER" ]; then
    echo "Error: Missing required variables in .env file"
    exit 1
fi

# Execute SSH command
ssh -t -p $PORT $REMOTE_USER@$REMOTE_HOST