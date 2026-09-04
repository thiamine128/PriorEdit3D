#!/bin/bash

PROCESS_KEYWORD="lightning" 
USER_NAME=$(whoami)
echo "--- 🛡️  GPU Training Cleanup Utility ---"
PIDS=$(pgrep -u $USER_NAME -f $PROCESS_KEYWORD)
if [ -z "$PIDS" ]; then
    echo "✅ No active $PROCESS_KEYWORD processes found for user $USER_NAME."
else
    echo "⚠️  Found the following PIDs: $PIDS"
    ps -up $PIDS
    
    if [[ "$1" != "--force" ]]; then
        read -p "Do you want to kill these processes? (y/n): " confirm
        if [[ $confirm != [yY] ]]; then
            echo "❌ Cleanup cancelled."
            exit 1
        fi
    fi

    echo "🚀 Killing processes..."
    kill -9 $PIDS
    sleep 2
fi