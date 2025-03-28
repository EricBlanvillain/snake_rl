# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Reorganized project structure: moved all Python files to `src` directory for better organization

### Planned
- Additional maze layouts
- Performance optimizations
- Extended evaluation metrics
- GUI improvements

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
