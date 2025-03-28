# evaluate.py
import pygame
import time
import os

# Important: Need to import algorithms to load models
from stable_baselines3 import PPO, DQN, A2C

from snake_env import SnakeEnv # We'll use the same env, but control both snakes
from constants import MODEL_DIR, MAZE_FILE, GAME_SPEED

# --- Evaluation Parameters ---
# Specify the paths to the two models you want to evaluate
MODEL_PATH_SNAKE1 = os.path.join(MODEL_DIR, "snake_approach_1_ppo.zip") # Example
MODEL_PATH_SNAKE2 = os.path.join(MODEL_DIR, "snake_approach_2_dqn.zip") # Example

# Determine which algorithm class was used for each model (based on filename or knowledge)
# This is crucial for loading correctly.
ALGO_SNAKE1 = PPO # Must match the algorithm used for training MODEL_PATH_SNAKE1
ALGO_SNAKE2 = DQN # Must match the algorithm used for training MODEL_PATH_SNAKE2

NUM_EPISODES = 5
RENDER_MODE = "human" # Use "human" to watch, None for faster eval (just print results)
DETERMINISTIC_PLAY = True # True: Agents pick best action, False: Sample action (more randomness)

# --- Load Models ---
if not os.path.exists(MODEL_PATH_SNAKE1) or not os.path.exists(MODEL_PATH_SNAKE2):
    print("Error: One or both model files not found!")
    print(f"Checked: {MODEL_PATH_SNAKE1}")
    print(f"Checked: {MODEL_PATH_SNAKE2}")
    print("Please train the models using train.py first.")
    exit()

print(f"Loading model for Snake 1 ({ALGO_SNAKE1.__name__}): {MODEL_PATH_SNAKE1}")
model1 = ALGO_SNAKE1.load(MODEL_PATH_SNAKE1)
print(f"Loading model for Snake 2 ({ALGO_SNAKE2.__name__}): {MODEL_PATH_SNAKE2}")
model2 = ALGO_SNAKE2.load(MODEL_PATH_SNAKE2)

# --- Setup Environment ---
# We need a way to get observations for *both* snakes and step using *both* actions.
# The existing SnakeEnv is designed for training one agent. We'll adapt its use here.
# We will get obs1, predict action1, get obs2 (by swapping snake roles in _get_obs logic implicitly), predict action2.

# Create a single env instance for rendering and game logic
eval_env = SnakeEnv(render_mode=RENDER_MODE, maze_file=MAZE_FILE, reward_approach=1) # Reward approach doesn't matter much here

# Helper function to get observation specifically for snake 2
# This involves temporarily swapping snake identities in the grid generation logic
def get_obs_for_snake2(env):
    # Create grid representation, but swap roles
    obs_grid = np.full((env.grid_height, env.grid_width), EMPTY, dtype=np.uint8)

    # Walls
    for (x, y) in env.maze.barriers:
        if 0 <= y < env.grid_height and 0 <= x < env.grid_width: obs_grid[y, x] = WALL
    # Food
    if env.food and 0 <= env.food.position.y < env.grid_height and 0 <= env.food.position.x < env.grid_width:
         obs_grid[env.food.position.y, env.food.position.x] = FOOD_ITEM
    # Powerup
    if env.powerup and 0 <= env.powerup.position.y < env.grid_height and 0 <= env.powerup.position.x < env.grid_width:
         obs_grid[env.powerup.position.y, env.powerup.position.x] = POWERUP_ITEM

    # Add snake 1 (as opponent)
    if env.snake1:
        for i, segment in enumerate(env.snake1.body):
             if 0 <= segment.y < env.grid_height and 0 <= segment.x < env.grid_width:
                obs_grid[segment.y, segment.x] = SNAKE2_HEAD if i == 0 else SNAKE2_BODY # Use opponent IDs

    # Add snake 2 (as self)
    if env.snake2:
        for i, segment in enumerate(env.snake2.body):
             if 0 <= segment.y < env.grid_height and 0 <= segment.x < env.grid_width:
                obs_grid[segment.y, segment.x] = SNAKE1_HEAD if i == 0 else SNAKE1_BODY # Use self IDs

    if OBS_TYPE == 'grid':
        # return obs_grid.flatten() # If models expect flattened input
        return obs_grid
    else:
        raise NotImplementedError("Vector obs for evaluation not fully implemented here")


# --- Evaluation Loop ---
total_scores = {1: 0, 2: 0}
wins = {1: 0, 2: 0}
draws = 0

for episode in range(NUM_EPISODES):
    print(f"\n--- Starting Evaluation Episode {episode + 1}/{NUM_EPISODES} ---")
    obs1, info = eval_env.reset()
    # Need the initial obs for snake 2 as well
    obs2 = get_obs_for_snake2(eval_env)

    terminated = False
    truncated = False
    episode_score1 = 0
    episode_score2 = 0
    step = 0

    while not terminated and not truncated:
        step += 1
        # Get actions from models
        action1, _ = model1.predict(obs1, deterministic=DETERMINISTIC_PLAY)
        action2, _ = model2.predict(obs2, deterministic=DETERMINISTIC_PLAY)

        # --- Need a modified step function or logic here ---
        # The env.step() expects only action1. We need to manually update based on both actions.
        # Re-implement the core logic from env.step(), but handling two agent actions.

        eval_env._current_step += 1 # Use internal counter if needed, or rely on loop var 'step'
        current_terminated = False
        current_truncated = eval_env._current_step >= MAX_STEPS_PER_EPISODE # Or use 'step'
        snake1_died_step = False
        snake2_died_step = False
        eval_env.snake1_death_reason = ""
        eval_env.snake2_death_reason = ""

        # Store original heads before move
        original_head1 = eval_env.snake1.head
        original_head2 = eval_env.snake2.head

        # Move snakes based on predicted actions
        eval_env.snake1.move(action1)
        eval_env.snake2.move(action2)
        next_head1 = eval_env.snake1.head
        next_head2 = eval_env.snake2.head

        # --- Check Collisions (adapted from env.step) ---
        if eval_env.maze.is_wall(next_head1.x, next_head1.y):
            snake1_died_step = True; eval_env.snake1_death_reason = "collision_wall"
        if eval_env.maze.is_wall(next_head2.x, next_head2.y):
            snake2_died_step = True; eval_env.snake2_death_reason = "collision_wall"

        if not snake1_died_step and eval_env.snake1.check_collision_self():
            snake1_died_step = True; eval_env.snake1_death_reason = "collision_self"
        if not snake2_died_step and eval_env.snake2.check_collision_self():
            snake2_died_step = True; eval_env.snake2_death_reason = "collision_self"

        if not snake1_died_step and not snake2_died_step and next_head1 == next_head2:
             snake1_died_step = True; eval_env.snake1_death_reason = "collision_opponent_head"
             snake2_died_step = True; eval_env.snake2_death_reason = "collision_opponent_head"

        snake1_body_set = set(eval_env.snake1.body[1:])
        snake2_body_set = set(eval_env.snake2.body[1:])

        if not snake1_died_step:
            if next_head1 in snake2_body_set or next_head2 == original_head1:
                 snake1_died_step = True; eval_env.snake1_death_reason = "collision_opponent_body"

        if not snake2_died_step:
             if next_head2 in snake1_body_set or next_head1 == original_head2:
                  snake2_died_step = True; eval_env.snake2_death_reason = "collision_opponent_body"

        # --- Handle Food Eating ---
        food_eaten_this_step = False
        if not snake1_died_step and eval_env.food and next_head1 == eval_env.food.position:
            eval_env.snake1.score += eval_env.food.points
            eval_env.snake1.grow()
            food_eaten_this_step = True
        # Check if snake 2 ate *different* food spot (unlikely but possible if food respawned)
        elif not snake2_died_step and eval_env.food and next_head2 == eval_env.food.position:
            eval_env.snake2.score += eval_env.food.points
            eval_env.snake2.grow()
            food_eaten_this_step = True

        # Place new food if eaten
        if food_eaten_this_step:
             eval_env.food = eval_env._place_item(Food)
             # Avoid instant overlap with snake heads after respawn
             if eval_env.food and eval_env.food.position == eval_env.snake1.head:
                 eval_env.snake1.score += eval_env.food.points # Give points anyway? Or re-place? Re-place is safer.
                 eval_env.food = eval_env._place_item(Food)
             if eval_env.food and eval_env.food.position == eval_env.snake2.head:
                 eval_env.snake2.score += eval_env.food.points
                 eval_env.food = eval_env._place_item(Food)


        # --- Handle PowerUp Eating ---
        powerup_eaten_this_step = False
        if not snake1_died_step and eval_env.powerup and next_head1 == eval_env.powerup.position:
             if eval_env.powerup.type == 'extra_points': eval_env.snake1.score += eval_env.powerup.points
             powerup_eaten_this_step = True
        elif not snake2_died_step and eval_env.powerup and next_head2 == eval_env.powerup.position:
             if eval_env.powerup.type == 'extra_points': eval_env.snake2.score += eval_env.powerup.points
             powerup_eaten_this_step = True

        if powerup_eaten_this_step:
            eval_env.powerup = eval_env._place_item(PowerUp) if random.random() < 0.3 else None
            # Add similar overlap check as food if needed

        # --- Update Game State ---
        terminated = snake1_died_step or snake2_died_step
        truncated = current_truncated

        # Get next observations
        obs1 = eval_env._get_obs() # Observation for Snake 1
        obs2 = get_obs_for_snake2(eval_env) # Observation for Snake 2

        # Render the current frame
        if RENDER_MODE == "human":
            eval_env.render()
            # Add a small delay to make it watchable
            time.sleep(1.0 / GAME_SPEED) # Control speed here

        # Check for pygame quit event if rendering
        if RENDER_MODE == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("User quit.")
                    terminated = True # End episode if user closes window


        # Store final scores for the episode before loop ends
        episode_score1 = eval_env.snake1.score
        episode_score2 = eval_env.snake2.score


    # --- Episode End ---
    print(f"Episode {episode + 1} finished after {step} steps.")
    print(f"  Snake 1 ({ALGO_SNAKE1.__name__}): Score={episode_score1}, Died={snake1_died_step} ({eval_env.snake1_death_reason})")
    print(f"  Snake 2 ({ALGO_SNAKE2.__name__}): Score={episode_score2}, Died={snake2_died_step} ({eval_env.snake2_death_reason})")

    total_scores[1] += episode_score1
    total_scores[2] += episode_score2

    # Determine winner
    if snake1_died_step and snake2_died_step:
        print("  Result: Draw (Both died)")
        # Optional: Score difference decides draw winner?
        if episode_score1 > episode_score2: wins[1] += 1
        elif episode_score2 > episode_score1: wins[2] += 1
        else: draws += 1
    elif snake1_died_step:
        print(f"  Result: Snake 2 ({ALGO_SNAKE2.__name__}) Wins!")
        wins[2] += 1
    elif snake2_died_step:
        print(f"  Result: Snake 1 ({ALGO_SNAKE1.__name__}) Wins!")
        wins[1] += 1
    elif truncated: # Neither died, game ended by steps
        print("  Result: Truncated (Max steps reached)")
        if episode_score1 > episode_score2:
            print(f"    Snake 1 ({ALGO_SNAKE1.__name__}) wins on score!")
            wins[1] += 1
        elif episode_score2 > episode_score1:
            print(f"    Snake 2 ({ALGO_SNAKE2.__name__}) wins on score!")
            wins[2] += 1
        else:
            print("    Draw on score!")
            draws += 1

eval_env.close()

# --- Final Results ---
print("\n\n--- Evaluation Summary ---")
print(f"Episodes Played: {NUM_EPISODES}")
print(f"Model 1 (Snake 1 - {ALGO_SNAKE1.__name__}):")
print(f"  Wins: {wins[1]}")
print(f"  Average Score: {total_scores[1] / NUM_EPISODES:.2f}")
print(f"Model 2 (Snake 2 - {ALGO_SNAKE2.__name__}):")
print(f"  Wins: {wins[2]}")
print(f"  Average Score: {total_scores[2] / NUM_EPISODES:.2f}")
print(f"Draws: {draws}")
print("-" * 26)

if wins[1] > wins[2]:
    print(f"Overall Winner: Model 1 ({ALGO_SNAKE1.__name__})")
elif wins[2] > wins[1]:
     print(f"Overall Winner: Model 2 ({ALGO_SNAKE2.__name__})")
else:
    # Decide tie based on score if wins are equal
    if total_scores[1] > total_scores[2]:
         print(f"Overall Winner: Model 1 ({ALGO_SNAKE1.__name__}) (Tie in wins, higher avg score)")
    elif total_scores[2] > total_scores[1]:
         print(f"Overall Winner: Model 2 ({ALGO_SNAKE2.__name__}) (Tie in wins, higher avg score)")
    else:
        print("Overall Result: Exact Tie!")
