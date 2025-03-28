# constants.py
import pygame
import torch

# --- Display and Game Settings ---
GRID_WIDTH = 19
GRID_HEIGHT = 15
BLOCK_SIZE = 30
GAME_SPEED = 60  # Frames per second

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN1 = (0, 255, 0)
BLUE1 = (0, 0, 255)
GREEN2 = (0, 200, 0)
BLUE2 = (0, 0, 200)
YELLOW = (255, 255, 0)

# --- Grid Cell Types ---
EMPTY = 0
WALL = 1
FOOD_ITEM = 2
POWERUP_ITEM = 3
SNAKE1_HEAD = 4
SNAKE1_BODY = 5
SNAKE2_HEAD = 6
SNAKE2_BODY = 7

# --- Observation Settings ---
OBS_TYPE = 'grid'  # 'grid' or 'vector'
# New: Additional observation features
USE_DISTANCE_FEATURES = True  # Include distance-based features
USE_DANGER_FEATURES = True    # Include danger detection features
USE_FOOD_DIRECTION = True     # Include food direction features

# --- Training Settings ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs"
MODEL_DIR = "models"

# --- Reward Structure - Approach 1 (Simple) ---
REWARD_STEP_A1 = -0.01      # Small penalty for each step to encourage efficiency
REWARD_FOOD_A1 = 10.0       # Reward for eating food
REWARD_POWERUP_A1 = 5.0     # Reward for getting powerup
REWARD_DEATH_A1 = -10.0     # Penalty for dying

# --- Reward Structure - Approach 2 (Sophisticated) ---
# Base rewards
REWARD_STEP_A2 = 0.1        # Small reward for surviving each step
REWARD_FOOD_A2 = 15.0       # Major reward for eating food
REWARD_POWERUP_A2 = 7.5     # Good reward for powerup
REWARD_DEATH_A2 = -15.0     # Significant death penalty

# Progressive rewards
REWARD_CLOSER_TO_FOOD = 0.2      # Original reward for moving closer to food
REWARD_FURTHER_FROM_FOOD = -0.1   # Original penalty for moving away from food
REWARD_EFFICIENT_PATH = 0.3       # Reward for moving along efficient path to food
REWARD_SURVIVAL_BONUS = 0.5       # Bonus for surviving longer than median episode length
REWARD_KILL_OPPONENT = 5.0        # Reward for eliminating opponent
REWARD_DEATH_BY_OPPONENT = -5.0   # Additional penalty for dying to opponent

# --- DQN Specific Settings ---
REPLAY_BUFFER_SIZE = 100_000
BATCH_SIZE = 128          # Standard batch size that worked well
LEARNING_RATE = 3e-4      # Original learning rate that showed good progress
GAMMA = 0.99             # High discount factor for long-term rewards
TAU = 0.005             # Soft update parameter
TRAIN_FREQ = 4          # Update frequency
GRADIENT_STEPS = 1       # Single gradient step per update
LEARNING_STARTS = 10_000  # Original learning start step
EXPLORATION_FRACTION = 0.2  # Original exploration fraction
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05

# --- Environment Settings ---
MAX_STEPS_PER_EPISODE = GRID_WIDTH * GRID_HEIGHT * 2  # Maximum steps before episode truncation
MAX_EPISODE_STEPS = 1000
POWERUP_CHANCE = 0.3       # Probability of powerup spawning
MAZE_ROTATION = True       # Whether to randomly rotate mazes during training
