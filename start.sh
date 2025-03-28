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
        tensorboard --logdir=runs/
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
