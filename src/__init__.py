"""
Snake Reinforcement Learning package.
This package contains the implementation of a Snake game environment and DQN agent.
"""

from .snake_env import SnakeEnv
from .game_elements import Point, Direction, Snake, Food, PowerUp
from .maze import Maze

__version__ = '1.0.0'
