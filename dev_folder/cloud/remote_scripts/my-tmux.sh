#!/bin/bash

export TZ=America/Los_Angeles
SESSION_NAME="arc_training"

# Check if the session exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Attaching to existing session: $SESSION_NAME"
    tmux attach-session -t "$SESSION_NAME"
else
    echo "Creating new session: $SESSION_NAME"
    tmux new-session -d -s "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "export TZ=America/Los_Angeles" C-m
    tmux attach-session -t "$SESSION_NAME"
fi