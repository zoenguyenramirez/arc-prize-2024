#!/bin/bash

# File: kill_process.sh
PGID_FILE=".TASK_PGID"

if [ ! -f "$PGID_FILE" ]; then
    echo "PGID file $PGID_FILE does not exist."
    exit 1
fi

PGID=$(cat "$PGID_FILE")

# Kill the main process and all its children
kill -- -$PGID

# Wait a moment to ensure all processes are terminated
sleep 1

# Check if the process and its children are really gone
if ! ps -p $PGID > /dev/null 2>&1 && ! pgrep -P $PGID > /dev/null 2>&1; then
    echo "Process with PGID $PGID and all its sub-processes have been killed."
    rm "$PGID_FILE"
else
    echo "Failed to kill all processes. Some may still be running."
    exit 1
fi