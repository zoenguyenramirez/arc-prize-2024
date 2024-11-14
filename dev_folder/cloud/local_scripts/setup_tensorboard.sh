#!/bin/bash

# Check if the local port argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide the local port as an argument."
    echo "Usage: $0 <local_port>"
    exit 1
fi

LOCAL_PORT=$1

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

echo "Setting up SSH port forwarding for TensorBoard..."
echo "TensorBoard will be accessible at http://localhost:${LOCAL_PORT} in your local browser"
echo "You will now enter an interactive SSH session."
echo "To exit the session and stop port forwarding, type 'exit' or press Ctrl+D."
echo "-----------------------------------------------------------"

# Set up SSH port forwarding
ssh -p $PORT -L ${LOCAL_PORT}:127.0.0.1:6006 $REMOTE_USER@$REMOTE_HOST

echo "-----------------------------------------------------------"
echo "SSH session ended. Port forwarding has been stopped."
echo "TensorBoard is no longer accessible."