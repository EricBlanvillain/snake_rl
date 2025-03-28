#!/bin/bash

# Function to display usage
show_usage() {
    echo "Usage: ./start.sh [option]"
    echo "Options:"
    echo "  game        - Start the snake game"
    echo "  train       - Start training the model"
    echo "  tensorboard - Start TensorBoard to view training metrics"
    echo "  help        - Show this help message"
}

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Check if an argument was provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Process the command line argument
case "$1" in
    "game")
        echo "Starting the snake game..."
        python src/visualize.py
        ;;
    "train")
        echo "Starting model training..."
        python src/train.py
        ;;
    "tensorboard")
        echo "Starting TensorBoard..."
        pkill tensorboard || true  # Kill any existing tensorboard processes
        sleep 1
        # Clean up any existing event files in /tmp
        rm -rf /tmp/tensorboard_* || true
        # Start tensorboard with more aggressive reload settings
        tensorboard --logdir=runs --reload_multifile=true --reload_interval=1 --samples_per_plugin=scalar=0 --purge_orphaned_data=true --max_reload_threads=8 --bind_all --load_fast=false
        ;;
    "help")
        show_usage
        ;;
    *)
        echo "Invalid option: $1"
        show_usage
        exit 1
        ;;
esac
