# Snake Reinforcement Learning

A Python implementation of the classic Snake game with a reinforcement learning agent. The agent learns to play Snake using Deep Q-Learning (DQN) in various maze environments, featuring a retro-punk visual style and two-player mode.

## Project Overview

This project implements:
- A customizable Snake game environment with retro-punk visuals
- Deep Q-Learning (DQN) agent for training
- Support for multiple maze configurations
- Two-player mode with AI opponent
- Interactive pause menu and game controls
- Leaderboard system
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
│   ├── maze.py          # Maze generation and handling
│   ├── visualize.py     # Game visualization and UI
│   ├── button.py        # UI button components
│   ├── menu.py          # Game menus
│   └── leaderboard.py   # Leaderboard system
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

## Game Features

### Controls
- **Spacebar/ESC**: Pause/Unpause game
- **L**: Toggle leaderboard (when game is over)
- **M**: Open maze selection menu
- Mouse control for menu navigation

### Game Modes
- Single player vs AI opponent
- Multiple AI difficulty levels for opponent
- Various maze layouts to choose from

### Visual Features
- Retro-punk aesthetic with neon effects
- Dynamic score display
- Death messages and game over screen
- Interactive pause menu
- Maze selection interface
- Leaderboard display

## Training

To train a new model:

```bash
python src/train.py
```

The training process includes:
- Automatic checkpoint saving
- TensorBoard logging
- Multiple training runs support
- Customizable reward structures

Training metrics are saved in the `runs/` directory and can be visualized using TensorBoard:

```bash
tensorboard --logdir=runs
```

## Evaluation

To evaluate a trained model:

```bash
python src/evaluate.py
```

The evaluation includes:
- Performance metrics
- Visualization of agent behavior
- Score tracking
- Death reason analysis

## Configuration

Key parameters can be adjusted in `src/constants.py`:
- Learning rate
- Discount factor
- Epsilon values for exploration
- Network architecture
- Reward structure
- Training episodes
- Game speed and visual settings
- Opponent AI behavior

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
- Compete effectively against the opponent snake

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a new branch for your feature
3. Submitting a pull request

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is open source and available under the MIT License.
