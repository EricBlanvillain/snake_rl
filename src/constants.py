# constants.py
import pygame
import torch

# --- Display and Game Settings ---
GRID_WIDTH = 17
GRID_HEIGHT = 16
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

# Retro punk color scheme
COLORS = {
    'background': (0, 0, 20),      # Dark blue background
    'grid': (40, 40, 60),         # Subtle grid lines
    'snake1': (0, 255, 100),      # Neon green for snake 1
    'snake2': (255, 50, 50),      # Neon red for snake 2
    'food': (255, 255, 0),        # Yellow food
    'wall': (100, 0, 255),        # Purple walls
    'text': (0, 255, 200),        # Cyan text
    'score': (255, 50, 150)       # Pink score
}

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
REWARD_TIMEOUT_A1 = -15.0   # Severe penalty for dying from food timeout
REWARD_HUNGER_A1 = -0.05    # Additional penalty when steps_since_last_food > 45 (75% of timeout)

# --- Reward Structure - Approach 2 (Sophisticated) ---
# Base rewards
REWARD_STEP_A2 = 0.05  # Increased step reward to strongly encourage survival
REWARD_FOOD_A2 = 20.0  # Increased food reward
REWARD_POWERUP_A2 = 10.0  # Increased powerup reward
REWARD_DEATH_A2 = -2.0  # Further reduced death penalty
REWARD_TIMEOUT_A2 = -1.0  # Reduced timeout penalty
REWARD_HUNGER_A2 = -0.005  # Minimal hunger penalty

# Progressive rewards
REWARD_CLOSER_TO_FOOD = 0.3  # Increased reward for moving towards food
REWARD_FURTHER_FROM_FOOD = -0.01  # Minimal penalty for moving away
REWARD_EFFICIENT_PATH = 5.0  # Increased efficient path reward significantly
REWARD_SURVIVAL_BONUS = 0.5  # Increased survival bonus
REWARD_KILL_OPPONENT = 5.0
REWARD_DEATH_BY_OPPONENT = -1.0

# Distance-based rewards
REWARD_WALL_DISTANCE = 0.2  # Increased wall distance reward
REWARD_OPPONENT_DISTANCE = 0.2  # Increased opponent distance reward

# --- DQN Specific Settings ---
REPLAY_BUFFER_SIZE = 100_000
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
TRAIN_FREQ = 4
GRADIENT_STEPS = 1
LEARNING_STARTS = 5_000  # Reduced to start learning earlier
EXPLORATION_FRACTION = 0.3  # Increased exploration period
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05

# --- Curriculum Learning Settings ---
CURRICULUM_ENABLED = True
CURRICULUM_STAGES = {
    1: {
        'maze_file': 'mazes/maze_empty.txt',
        'min_score_advance': 1,  # Even easier to advance
        'opponent_enabled': False,
        'powerups_enabled': False,
        'food_max_distance': 3,  # Keep food extremely close initially
    },
    2: {
        'maze_file': 'mazes/maze_empty.txt',
        'min_score_advance': 2,
        'opponent_enabled': False,
        'powerups_enabled': False,
        'food_max_distance': 5,
    },
    3: {
        'maze_file': 'mazes/maze_empty.txt',  # Still empty maze
        'min_score_advance': 3,
        'opponent_enabled': False,
        'powerups_enabled': True,
        'food_max_distance': 8,
    },
    4: {
        'maze_file': 'mazes/maze_simple.txt',  # Now introduce simple maze
        'min_score_advance': 4,
        'opponent_enabled': False,
        'powerups_enabled': True,
        'food_max_distance': 10,
    },
    5: {
        'maze_file': 'mazes/maze_medium.txt',
        'min_score_advance': 5,
        'opponent_enabled': True,
        'powerups_enabled': True,
        'food_max_distance': None,
    }
}

# --- Environment Settings ---
MAX_STEPS_PER_EPISODE = GRID_WIDTH * GRID_HEIGHT * 2
MAX_EPISODE_STEPS = 1000
POWERUP_SPAWN_CHANCE = 0.2  # Further reduced to simplify early learning
MAZE_ROTATION = False  # Disabled to make early learning easier
FOOD_TIMEOUT = 150  # Increased food timeout significantly

# --- Additional Observation Features ---
USE_DISTANCE_FEATURES = True      # Include distance-based features
USE_DANGER_FEATURES = True        # Include danger detection features
USE_FOOD_DIRECTION = True         # Include food direction features
USE_LENGTH_FEATURE = True         # Include normalized snake length
USE_WALL_DISTANCE = True          # Include distance to nearest wall
USE_OPPONENT_FEATURES = True      # Include opponent-related features
USE_POWERUP_FEATURES = True       # Include powerup-related features
