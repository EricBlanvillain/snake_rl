# Snake Reinforcement Learning

A Python implementation of the classic Snake game with a reinforcement learning agent. The agent learns to play Snake using Deep Q-Learning (DQN) in various maze environments.

## Project Overview

This project implements:
- A customizable Snake game environment
- Deep Q-Learning (DQN) agent for training
- Support for multiple maze configurations
- TensorBoard integration for training visualization
- Evaluation tools for trained models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EricBlanvillain/snake_rl.git
cd snake_rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
snake_rl/
├── src/                    # Source code directory
│   ├── snake_env.py       # Custom OpenAI Gym environment
│   ├── game_elements.py   # Core game mechanics and objects
│   ├── train.py          # Training script for DQN agent
│   ├── evaluate.py       # Evaluation script for models
│   ├── constants.py      # Configuration parameters
│   └── maze.py          # Maze generation and handling
├── mazes/                 # Maze configuration files
│   ├── default.txt       # Default empty maze
│   ├── maze1.txt        # Various maze layouts
│   ├── maze2.txt
│   └── maze3.txt
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── CHANGELOG.md          # Version history
├── LICENSE              # MIT License
└── CONTRIBUTING.md      # Contributing guidelines
```

## Training

To train a new model:

```bash
python src/train.py
```

The training process includes:
- Automatic checkpoint saving
- TensorBoard logging
- Multiple training runs support

Training metrics are saved in the `runs/` directory and can be visualized using TensorBoard:

```bash
tensorboard --logdir=runs
```

## Evaluation

To evaluate a trained model:

```bash
python src/evaluate.py
```

## Configuration

Key parameters can be adjusted in `src/constants.py`:
- Learning rate
- Discount factor
- Epsilon values for exploration
- Network architecture
- Reward structure
- Training episodes

## Maze Configuration

Custom mazes can be created by adding new text files to the `mazes/` directory. The maze format uses:
- `#`: Wall
- ` `: Empty space
- `S`: Starting position (optional)

## Performance

The agent typically learns to:
- Navigate effectively through the maze
- Collect food efficiently
- Avoid collisions with walls and itself
- Develop strategies for different maze layouts

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a new branch for your feature
3. Submitting a pull request

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is open source and available under the MIT License.
