# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Interactive pause menu with spacebar/ESC control
- Two-player mode with AI opponent
- Retro-punk visual style with neon effects
- Leaderboard system for tracking high scores
- Death messages and game over screen
- Maze selection interface
- Dynamic score display
- Multiple AI difficulty levels for opponent

### Changed
- Reorganized project structure: moved all Python files to `src` directory for better organization
- Enhanced visualization system with retro-punk aesthetic
- Improved game controls and user interface
- Added proper game state management for pausing and resuming
- Updated snake movement and collision detection

### Fixed
- Snake movement validation and control issues
- Game termination logic for both snakes
- Action validation for AI opponent
- Collision detection between snakes

### Planned
- Additional maze layouts
- Performance optimizations
- Extended evaluation metrics
- GUI improvements
- Online multiplayer support
- Custom snake skins
- Achievement system

## [1.0.0] - 2024-03-28

### Added
- Initial release of Snake RL project
- Custom OpenAI Gym environment for Snake game
- Deep Q-Learning (DQN) implementation
- Multiple maze configurations support
- TensorBoard integration for training visualization
- Evaluation tools for trained models
- Configurable reward structure and state features
- Multiple training runs support with separate logging
- Automated checkpoint saving system

### Changed
- Optimized reward structure for better agent performance
- Refined state features for improved learning
- Adjusted training parameters based on experimental results

### Technical Details
- Implemented custom Snake game environment in `src/snake_env.py`
- Created modular game mechanics in `src/game_elements.py`
- Added maze generation and handling in `src/maze.py`
- Developed training script with configurable parameters in `src/train.py`
- Added evaluation capabilities in `src/evaluate.py`
- Centralized configuration in `src/constants.py`

[1.0.0]: https://github.com/EricBlanvillain/snake_rl/releases/tag/v1.0.0
[Unreleased]: https://github.com/EricBlanvillain/snake_rl/compare/v1.0.0...HEAD
