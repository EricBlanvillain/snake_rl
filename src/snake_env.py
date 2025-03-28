# snake_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame # Only needed for render method
import random
import os

from constants import *
from maze import Maze
from game_elements import Point, Direction, Snake, Food, PowerUp

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": GAME_SPEED}

    def __init__(self, render_mode=None, maze_file="mazes/maze_natural.txt", reward_approach=1, opponent_policy='stay_still'):
        super().__init__()

        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.block_size = BLOCK_SIZE
        self.window_width = self.grid_width * self.block_size
        self.window_height = self.grid_height * self.block_size

        self.maze = Maze(maze_file)
        self.reward_approach = reward_approach
        self.opponent_policy_type = opponent_policy

        # Initialize game elements
        self.food = None
        self.powerup = None
        self.snake1 = None
        self.snake2 = None
        self.snake1_death_reason = ""
        self.snake2_death_reason = ""

        # Calculate total observation space size
        self.grid_size = self.grid_width * self.grid_height
        n_additional_features = 0
        if USE_DISTANCE_FEATURES:
            n_additional_features += 3  # food, wall, opponent distances
        if USE_DANGER_FEATURES:
            n_additional_features += 4  # danger in 4 directions
        if USE_FOOD_DIRECTION:
            n_additional_features += 4  # food direction one-hot

        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=max(SNAKE1_HEAD, SNAKE1_BODY, SNAKE2_HEAD, SNAKE2_BODY),
            shape=(self.grid_size + n_additional_features,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Snake RL")
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()

        self._current_step = 0
        self._median_episode_length = 10  # Initial estimate, will be updated during training

    def _get_reward(self, snake1_ate_food, snake1_ate_powerup, snake1_died, snake2_died, info):
        if self.reward_approach == 1:
            reward = REWARD_STEP_A1
            if snake1_ate_food:
                reward += REWARD_FOOD_A1
            if snake1_ate_powerup:
                reward += REWARD_POWERUP_A1
            if snake1_died:
                reward += REWARD_DEATH_A1
        elif self.reward_approach == 2:
            reward = REWARD_STEP_A2  # Base survival reward

            # Food-related rewards
            if snake1_ate_food:
                reward += REWARD_FOOD_A2
            else:
                # Calculate distance-based rewards
                old_dist = self._prev_food_distance if hasattr(self, '_prev_food_distance') else None
                new_dist = self._get_manhattan_distance(self.snake1.body[0], self.food.position)
                self._prev_food_distance = new_dist

                if old_dist is not None:
                    if new_dist < old_dist:
                        reward += REWARD_CLOSER_TO_FOOD
                    elif new_dist > old_dist:
                        reward += REWARD_FURTHER_FROM_FOOD

            # Powerup rewards
            if snake1_ate_powerup:
                reward += REWARD_POWERUP_A2

            # Death penalties
            if snake1_died:
                reward += REWARD_DEATH_A2
                if info.get("snake1_death_reason") == "collision_opponent":
                    reward += REWARD_DEATH_BY_OPPONENT

            # Opponent interaction
            if snake2_died and not snake1_died:
                reward += REWARD_KILL_OPPONENT

            # Survival bonus
            if self._current_step > self._median_episode_length:
                reward += REWARD_SURVIVAL_BONUS
        else:
            raise ValueError(f"Unknown reward approach: {self.reward_approach}")

        return reward

    def _get_obs(self):
        # Create base grid representation
        obs_grid = np.full((self.grid_height, self.grid_width), EMPTY, dtype=np.float32)

        # Add walls
        for (x, y) in self.maze.barriers:
            if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                obs_grid[y, x] = WALL

        # Add food
        if self.food and 0 <= self.food.position.y < self.grid_height and 0 <= self.food.position.x < self.grid_width:
            obs_grid[self.food.position.y, self.food.position.x] = FOOD_ITEM

        # Add powerup
        if self.powerup and 0 <= self.powerup.position.y < self.grid_height and 0 <= self.powerup.position.x < self.grid_width:
            obs_grid[self.powerup.position.y, self.powerup.position.x] = POWERUP_ITEM

        # Add snake 2 (opponent)
        if self.snake2:
            for i, segment in enumerate(self.snake2.body):
                if 0 <= segment.y < self.grid_height and 0 <= segment.x < self.grid_width:
                    obs_grid[segment.y, segment.x] = SNAKE2_HEAD if i == 0 else SNAKE2_BODY

        # Add snake 1 (agent)
        if self.snake1:
            for i, segment in enumerate(self.snake1.body):
                if 0 <= segment.y < self.grid_height and 0 <= segment.x < self.grid_width:
                    obs_grid[segment.y, segment.x] = SNAKE1_HEAD if i == 0 else SNAKE1_BODY

        # Flatten the grid
        flat_grid = obs_grid.flatten()

        # Calculate additional features
        additional_features = []

        if USE_DISTANCE_FEATURES:
            # Distance to food
            food_dist = self._get_manhattan_distance(self.snake1.body[0], self.food.position)
            # Distance to nearest wall
            wall_dist = self._get_nearest_wall_distance(self.snake1.body[0])
            # Distance to opponent head
            opp_dist = self._get_manhattan_distance(self.snake1.body[0], self.snake2.body[0])
            additional_features.extend([food_dist/self.grid_width, wall_dist/self.grid_width, opp_dist/self.grid_width])

        if USE_DANGER_FEATURES:
            # Check immediate danger in each direction
            dangers = self._get_danger_in_directions()
            additional_features.extend(dangers)

        if USE_FOOD_DIRECTION:
            # Relative direction to food
            food_dir = self._get_relative_direction(self.snake1.body[0], self.food.position)
            additional_features.extend(food_dir)

        # Combine grid with additional features
        return np.concatenate([flat_grid, np.array(additional_features, dtype=np.float32)])

    def _get_manhattan_distance(self, point1, point2):
        return abs(point1.x - point2.x) + abs(point1.y - point2.y)

    def _get_nearest_wall_distance(self, point):
        distances = []
        for wall in self.maze.barriers:
            dist = self._get_manhattan_distance(point, Point(wall[0], wall[1]))
            distances.append(dist)
        return min(distances) if distances else self.grid_width

    def _get_danger_in_directions(self):
        head = self.snake1.body[0]
        dangers = []
        for direction in Direction.INDEX_TO_DIR:
            dx, dy = Direction.get_components(direction)
            next_pos = Point(head.x + dx, head.y + dy)
            # Check if next position would result in collision
            is_danger = (
                self.maze.is_wall(next_pos.x, next_pos.y) or
                next_pos in self.snake1.body[1:] or
                next_pos in self.snake2.body
            )
            dangers.append(float(is_danger))
        return dangers

    def _get_relative_direction(self, from_point, to_point):
        dx = to_point.x - from_point.x
        dy = to_point.y - from_point.y
        direction = [0, 0, 0, 0]  # up, down, left, right

        if abs(dx) > abs(dy):
            direction[2 if dx < 0 else 3] = 1  # left or right
        else:
            direction[0 if dy < 0 else 1] = 1  # up or down

        return direction

    def _get_info(self):
        return {
            "snake1_score": self.snake1.score,
            "snake2_score": self.snake2.score,
            "steps": self._current_step,
            "snake1_len": len(self.snake1.body),
            "snake2_len": len(self.snake2.body),
            "snake1_death_reason": self.snake1_death_reason,
            "snake2_death_reason": self.snake2_death_reason,
        }

    def _place_item(self, item_class, **kwargs):
        occupied = self.snake1.get_positions() | self.snake2.get_positions()
        if self.food: occupied.add(self.food.position)
        if self.powerup: occupied.add(self.powerup.position)

        pos = self.maze.get_random_empty_cell(occupied)
        return item_class(pos, **kwargs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self._current_step = 0
        self.snake1_death_reason = ""
        self.snake2_death_reason = ""

        # Only select random maze if no specific maze is loaded
        if not self.maze or not self.maze.is_loaded:
            maze_files = [f for f in os.listdir("mazes") if f.endswith(".txt")]
            if maze_files:
                selected_maze = random.choice(maze_files)
                new_maze_path = os.path.join("mazes", selected_maze)
                if new_maze_path != getattr(self.maze, 'filepath', None):
                    self.maze = Maze(new_maze_path)
                    if MAZE_ROTATION and random.random() < 0.5:
                        self.maze.rotate_180()

        # Place snakes (ensure they don't start on walls or overlap)
        while True:
            start_x1 = random.randint(1, self.grid_width - 2)
            start_y1 = random.randint(1, self.grid_height - 2)
            if not self.maze.is_wall(start_x1, start_y1):
                break
        start_dir1 = random.choice(list(Direction.INDEX_TO_DIR))
        self.snake1 = Snake(1, (start_x1, start_y1), start_dir1, GREEN1, BLUE1)

        while True:
            start_x2 = random.randint(1, self.grid_width - 2)
            start_y2 = random.randint(1, self.grid_height - 2)
            # Ensure not wall and not too close to snake1 start
            if not self.maze.is_wall(start_x2, start_y2) and \
               (abs(start_x1 - start_x2) + abs(start_y1 - start_y2)) > 3: # Min distance
                 if Point(start_x2, start_y2) not in self.snake1.get_positions(): # Avoid initial overlap
                    break
        start_dir2 = random.choice(list(Direction.INDEX_TO_DIR))
        self.snake2 = Snake(2, (start_x2, start_y2), start_dir2, GREEN2, BLUE2)

        # Place initial food and powerup
        self.food = self._place_item(Food)
        self.powerup = self._place_item(PowerUp) if random.random() < 0.5 else None # Chance to have powerup

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_opponent_action(self):
        """Determine Snake 2's action based on its policy."""
        # If snake is dead, return action that maintains current direction
        if self.snake2_death_reason:
            try:
                current_dir_index = Direction.INDEX_TO_DIR.index(self.snake2.direction)
                return current_dir_index
            except ValueError:
                return 0  # Default to UP if direction is invalid

        if self.opponent_policy_type == 'stay_still':
            # Find action index that corresponds to current direction
            try:
                current_dir_index = Direction.INDEX_TO_DIR.index(self.snake2.direction)
                return current_dir_index
            except ValueError:
                return random.randint(0, 3) # Fallback if direction is weird

        elif self.opponent_policy_type == 'random':
            # Avoid immediate reversal if possible
            possible_actions = list(range(4))
            opposite_dir = Direction.OPPOSITE.get(self.snake2.direction)
            if opposite_dir:
                try:
                    opposite_action_index = Direction.INDEX_TO_DIR.index(opposite_dir)
                    if len(self.snake2.body)>1 : # only prevent reversal if snake is longer than 1
                         possible_actions.remove(opposite_action_index)
                except (ValueError, IndexError):
                    pass # Should not happen
            return random.choice(possible_actions) if possible_actions else random.randint(0,3)

        elif self.opponent_policy_type == 'basic_follow':
             # Very simple: move towards food if possible, else move randomly avoiding walls/self
            head = self.snake2.head
            target = self.food.position if self.food else None

            best_action = -1
            min_dist = float('inf')

            possible_actions = list(range(4))
             # Remove reverse action
            opposite_dir = Direction.OPPOSITE.get(self.snake2.direction)
            if opposite_dir and len(self.snake2.body) > 1:
                try:
                    opposite_action_index = Direction.INDEX_TO_DIR.index(opposite_dir)
                    possible_actions.remove(opposite_action_index)
                except ValueError: pass

            valid_actions = []
            for action in possible_actions:
                direction = Direction.INDEX_TO_DIR[action]
                next_pos = Point(head.x + direction.x, head.y + direction.y)

                # Check for immediate death collisions (wall or self)
                if not self.maze.is_wall(next_pos.x, next_pos.y) and \
                   next_pos not in self.snake2.body[1:]: # Don't check head collision yet
                    valid_actions.append(action)

                    # If food exists, check distance
                    if target:
                        dist = abs(next_pos.x - target.x) + abs(next_pos.y - target.y)
                        if dist < min_dist:
                            min_dist = dist
                            best_action = action

            if best_action != -1: # Found a move towards food
                return best_action
            elif valid_actions: # No food path or no food, pick random valid move
                return random.choice(valid_actions)
            else: # No valid moves, just pick original random (will likely die)
                 return random.randint(0, 3)
        else:
            return random.randint(0, 3) # Default random action

    def step(self, action):
        # Execute one time step within the environment
        self._current_step += 1

        # Move snakes (only if they're alive)
        if not self.snake1_death_reason:  # Only move if snake1 is alive
            self.snake1.move(action)

        opponent_action = self._get_opponent_action()
        if not self.snake2_death_reason:  # Only move if snake2 is alive
            self.snake2.move(opponent_action)

        # Track what happened this step
        snake1_ate_food = False
        snake1_ate_powerup = False
        snake1_died = False
        snake2_died = False

        # Check collisions with walls (only for living snakes)
        if not self.snake1_death_reason:
            if self.maze.is_wall(self.snake1.head.x, self.snake1.head.y):
                snake1_died = True
                self.snake1_death_reason = "collision_wall"

        if not self.snake2_death_reason:
            if self.maze.is_wall(self.snake2.head.x, self.snake2.head.y):
                snake2_died = True
                self.snake2_death_reason = "collision_wall"

        # Check snake self-collisions (only for living snakes)
        if not self.snake1_death_reason:
            if self.snake1.check_collision_self():
                snake1_died = True
                self.snake1_death_reason = "collision_self"

        if not self.snake2_death_reason:
            if self.snake2.check_collision_self():
                snake2_died = True
                self.snake2_death_reason = "collision_self"

        # Check snake-snake collisions (only for living snakes)
        if not self.snake1_death_reason and not self.snake2_death_reason:
            if self.snake1.head in self.snake2.body:
                snake1_died = True
                self.snake1_death_reason = "collision_opponent"
            if self.snake2.head in self.snake1.body:
                snake2_died = True
                self.snake2_death_reason = "collision_opponent"

        # Check food collision (only for living snakes)
        if not snake1_died and not self.snake1_death_reason and self.snake1.head == self.food.position:
            snake1_ate_food = True
            self.snake1.grow()
            self.snake1.score += self.food.points
            self.food = self._place_item(Food)

        if not snake2_died and not self.snake2_death_reason and self.snake2.head == self.food.position:
            self.snake2.grow()
            self.snake2.score += self.food.points
            self.food = self._place_item(Food)

        # Check powerup collision (only for living snakes)
        if self.powerup:
            if not snake1_died and not self.snake1_death_reason and self.snake1.head == self.powerup.position:
                snake1_ate_powerup = True
                if self.powerup.type == 'extra_points':
                    self.snake1.score += self.powerup.points
                self.powerup = None
            elif not snake2_died and not self.snake2_death_reason and self.snake2.head == self.powerup.position:
                if self.powerup.type == 'extra_points':
                    self.snake2.score += self.powerup.points
                self.powerup = None

        # Terminate if both snakes are dead or max steps reached
        terminated = (snake1_died or bool(self.snake1_death_reason)) and (snake2_died or bool(self.snake2_death_reason))
        truncated = self._current_step >= MAX_STEPS_PER_EPISODE

        # Get reward
        reward = self._get_reward(snake1_ate_food, snake1_ate_powerup, snake1_died, snake2_died, {
            "snake1_death_reason": self.snake1_death_reason,
            "snake2_death_reason": self.snake2_death_reason
        })

        # Get observation
        observation = self._get_obs()

        # Get info
        info = self._get_info()
        info["snake1_died"] = snake1_died
        info["snake2_died"] = snake2_died
        info["snake1_death_reason"] = self.snake1_death_reason if snake1_died else ""
        info["snake2_death_reason"] = self.snake2_death_reason if snake2_died else ""

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        # human mode rendering is handled in step/reset

    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Snake RL")
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
             self.clock = pygame.time.Clock()

        # Create a canvas (surface) to draw on
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill(WHITE)

        # Draw maze
        self.maze.draw(canvas, self.block_size)

        # Draw food
        if self.food:
            self.food.draw(canvas, self.block_size)

        # Draw powerup
        if self.powerup:
            self.powerup.draw(canvas, self.block_size)

        # Draw snakes
        if self.snake2: # Draw opponent first
             self.snake2.draw(canvas, self.block_size)
        if self.snake1: # Draw agent last
             self.snake1.draw(canvas, self.block_size)


        # Display Scores (Optional)
        font = pygame.font.SysFont('arial', 20)
        score_text1 = font.render(f"S1: {self.snake1.score}", True, GREEN1)
        score_text2 = font.render(f"S2: {self.snake2.score}", True, GREEN2)
        canvas.blit(score_text1, (5, 5))
        canvas.blit(score_text2, (self.window_width - score_text2.get_width() - 5, 5))

        if self.render_mode == "human":
            # Update the screen
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump() # Process Pygame events
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None # No return needed for human mode

        elif self.render_mode == "rgb_array":
             # Return pixels as numpy array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def _get_obs_for_snake2(self):
        """Get observation specifically for snake 2."""
        # Create grid representation, but swap roles
        obs_grid = np.full((self.grid_height, self.grid_width), EMPTY, dtype=np.float32)

        # Add walls
        for (x, y) in self.maze.barriers:
            if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                obs_grid[y, x] = WALL

        # Add food
        if self.food and 0 <= self.food.position.y < self.grid_height and 0 <= self.food.position.x < self.grid_width:
            obs_grid[self.food.position.y, self.food.position.x] = FOOD_ITEM

        # Add powerup
        if self.powerup and 0 <= self.powerup.position.y < self.grid_height and 0 <= self.powerup.position.x < self.grid_width:
            obs_grid[self.powerup.position.y, self.powerup.position.x] = POWERUP_ITEM

        # Add snake 1 (as opponent)
        if self.snake1:
            for i, segment in enumerate(self.snake1.body):
                if 0 <= segment.y < self.grid_height and 0 <= segment.x < self.grid_width:
                    obs_grid[segment.y, segment.x] = SNAKE2_HEAD if i == 0 else SNAKE2_BODY

        # Add snake 2 (as self)
        if self.snake2:
            for i, segment in enumerate(self.snake2.body):
                if 0 <= segment.y < self.grid_height and 0 <= segment.x < self.grid_width:
                    obs_grid[segment.y, segment.x] = SNAKE1_HEAD if i == 0 else SNAKE1_BODY

        # Flatten the grid
        flat_grid = obs_grid.flatten()

        # Calculate additional features
        additional_features = []

        if USE_DISTANCE_FEATURES:
            # Distance to food
            food_dist = self._get_manhattan_distance(self.snake2.body[0], self.food.position)
            # Distance to nearest wall
            wall_dist = self._get_nearest_wall_distance(self.snake2.body[0])
            # Distance to opponent head
            opp_dist = self._get_manhattan_distance(self.snake2.body[0], self.snake1.body[0])
            additional_features.extend([food_dist/self.grid_width, wall_dist/self.grid_width, opp_dist/self.grid_width])

        if USE_DANGER_FEATURES:
            # Check immediate danger in each direction
            dangers = self._get_danger_in_directions(for_snake2=True)
            additional_features.extend(dangers)

        if USE_FOOD_DIRECTION:
            # Relative direction to food
            food_dir = self._get_relative_direction(self.snake2.body[0], self.food.position)
            additional_features.extend(food_dir)

        # Combine grid with additional features
        return np.concatenate([flat_grid, np.array(additional_features, dtype=np.float32)])

    def _get_danger_in_directions(self, for_snake2=False):
        """Get danger in each direction for either snake."""
        snake = self.snake2 if for_snake2 else self.snake1
        other_snake = self.snake1 if for_snake2 else self.snake2
        head = snake.body[0]
        dangers = []

        for direction in Direction.INDEX_TO_DIR:
            dx, dy = Direction.get_components(direction)
            next_pos = Point(head.x + dx, head.y + dy)
            # Check if next position would result in collision
            is_danger = (
                self.maze.is_wall(next_pos.x, next_pos.y) or
                next_pos in snake.body[1:] or
                (other_snake and next_pos in other_snake.body)
            )
            dangers.append(float(is_danger))
        return dangers

    def update_maze(self, new_maze_file):
        """Update the current maze during training"""
        print(f"Loading maze from {new_maze_file}")

        # Don't reload maze if it's the same, but still reset game state
        if self.maze and self.maze.filepath == new_maze_file and self.maze.is_loaded:
            print(f"Maze {new_maze_file} is already loaded, resetting game state")
            try:
                # Reset game state
                self._current_step = 0
                self.snake1_death_reason = ""
                self.snake2_death_reason = ""

                # Place snakes in valid positions
                while True:
                    start_x1 = random.randint(1, self.grid_width - 2)
                    start_y1 = random.randint(1, self.grid_height - 2)
                    if not self.maze.is_wall(start_x1, start_y1):
                        break
                self.snake1 = Snake(1, (start_x1, start_y1), random.choice(list(Direction.INDEX_TO_DIR)), GREEN1, BLUE1)

                while True:
                    start_x2 = random.randint(1, self.grid_width - 2)
                    start_y2 = random.randint(1, self.grid_height - 2)
                    if not self.maze.is_wall(start_x2, start_y2) and \
                       (abs(start_x1 - start_x2) + abs(start_y1 - start_y2)) > 3 and \
                       Point(start_x2, start_y2) not in self.snake1.get_positions():
                        break
                self.snake2 = Snake(2, (start_x2, start_y2), random.choice(list(Direction.INDEX_TO_DIR)), GREEN2, BLUE2)

                # Place food and powerup
                self.food = self._place_item(Food)
                self.powerup = self._place_item(PowerUp) if random.random() < 0.5 else None

                print(f"Successfully reset game state with existing maze")
                return True
            except Exception as e:
                print(f"Error resetting game state: {str(e)}")
                return False

        # Validate the new maze file path
        if not os.path.exists(new_maze_file):
            print(f"Error: Maze file '{new_maze_file}' does not exist")
            return False

        # Create a new maze instance with the new file
        new_maze = Maze(new_maze_file)
        if new_maze.is_loaded:  # Only update if the maze loaded successfully
            # Store current maze as backup in case reset fails
            old_maze = self.maze
            old_width = self.grid_width
            old_height = self.grid_height

            try:
                self.maze = new_maze
                self.grid_width = self.maze.width
                self.grid_height = self.maze.height
                self.window_width = self.grid_width * self.block_size
                self.window_height = self.grid_height * self.block_size

                # Reset game state
                self._current_step = 0
                self.snake1_death_reason = ""
                self.snake2_death_reason = ""

                # Place snakes in valid positions
                while True:
                    start_x1 = random.randint(1, self.grid_width - 2)
                    start_y1 = random.randint(1, self.grid_height - 2)
                    if not self.maze.is_wall(start_x1, start_y1):
                        break
                self.snake1 = Snake(1, (start_x1, start_y1), random.choice(list(Direction.INDEX_TO_DIR)), GREEN1, BLUE1)

                while True:
                    start_x2 = random.randint(1, self.grid_width - 2)
                    start_y2 = random.randint(1, self.grid_height - 2)
                    if not self.maze.is_wall(start_x2, start_y2) and \
                       (abs(start_x1 - start_x2) + abs(start_y1 - start_y2)) > 3 and \
                       Point(start_x2, start_y2) not in self.snake1.get_positions():
                        break
                self.snake2 = Snake(2, (start_x2, start_y2), random.choice(list(Direction.INDEX_TO_DIR)), GREEN2, BLUE2)

                # Place food and powerup
                self.food = self._place_item(Food)
                self.powerup = self._place_item(PowerUp) if random.random() < 0.5 else None

                print(f"Successfully updated maze to {new_maze_file}")
                return True
            except Exception as e:
                # Restore old maze if reset fails
                print(f"Error during maze update: {str(e)}")
                self.maze = old_maze
                self.grid_width = old_width
                self.grid_height = old_height
                self.window_width = self.grid_width * self.block_size
                self.window_height = self.grid_height * self.block_size
                return False
        else:
            print(f"Failed to load maze from {new_maze_file}, keeping current maze")
            return False

# --- Basic Env Test ---
if __name__ == '__main__':
    # Create dummy maze if needed
    os.makedirs("mazes", exist_ok=True)
    if not os.path.exists("mazes/default.txt"):
         with open("mazes/default.txt", "w") as f:
             f.write("#" * GRID_WIDTH + "\n")
             for _ in range(GRID_HEIGHT - 2):
                 f.write("#" + " " * (GRID_WIDTH - 2) + "#\n")
             f.write("#" * GRID_WIDTH + "\n")

    # Test with human rendering and random actions
    # env = SnakeEnv(render_mode="human", maze_file="mazes/default.txt", reward_approach=1, opponent_policy='basic_follow')
    env = SnakeEnv(render_mode="human", maze_file="mazes/maze1.txt", reward_approach=2, opponent_policy='basic_follow') # Try a maze

    # Use check_env from stable_baselines3
    from stable_baselines3.common.env_checker import check_env
    # check_env(env) # Good practice to run this! It will likely complain if obs isn't flattened for MlpPolicy

    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not terminated and not truncated:
        action = env.action_space.sample()  # Take random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        # env.render() # Already called in step for human mode

        # Add a small delay to make it watchable
        # import time
        # time.sleep(0.05)

        if terminated or truncated:
             print("-" * 20)
             print(f"Episode Finished!")
             print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
             print(f"Steps: {step_count}")
             print(f"Final Info: {info}")
             print(f"Total Reward: {total_reward:.2f}")
             print("-" * 20)
             # Optionally reset for another episode
             # obs, info = env.reset()
             # terminated = False
             # truncated = False
             # total_reward = 0
             # step_count = 0


    env.close()
