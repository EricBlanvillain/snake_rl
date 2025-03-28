snake_rl/
├── constants.py         # Game settings, colors, sizes, RL params
├── maze.py              # Maze loading and representation
├── game_elements.py     # Snake, Food, PowerUp classes
├── snake_env.py         # The Gymnasium Environment for RL training
├── game_logic.py        # Core game logic (can be part of snake_env or separate)
├── train.py             # Script to train the RL agents (two approaches)
├── evaluate.py          # Script to run the trained agents against each other
├── utils.py             # Helper functions (optional)
├── mazes/
│   ├── default.txt
│   ├── maze1.txt
│   └── maze2.txt
├── models/              # Directory to save trained models
└── requirements.txt     # Project dependencies
