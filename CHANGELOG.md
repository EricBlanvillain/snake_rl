# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Implemented custom Snake game environment in `snake_env.py`
- Created modular game mechanics in `game_elements.py`
- Added maze generation and handling in `maze.py`
- Developed training script with configurable parameters in `train.py`
- Added evaluation capabilities in `evaluate.py`
- Centralized configuration in `constants.py`

## [Unreleased]
### Planned
- Additional maze layouts
- Performance optimizations
- Extended evaluation metrics
- GUI improvements

[1.0.0]: https://github.com/EricBlanvillain/snake_rl/releases/tag/v1.0.0
[Unreleased]: https://github.com/EricBlanvillain/snake_rl/compare/v1.0.0...HEAD
