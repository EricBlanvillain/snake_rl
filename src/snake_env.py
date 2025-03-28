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
            n_additional_features += 2  # relative position (x, y)
            n_additional_features += 2  # body and powerup distances
        if USE_DANGER_FEATURES:
            n_additional_features += 4  # immediate danger in 4 directions
            n_additional_features += 4  # future danger in 4 directions
        if USE_FOOD_DIRECTION:
            n_additional_features += 4  # food direction one-hot
            n_additional_features += 4  # current direction one-hot
        n_additional_features += 1  # normalized snake length

        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=max(SNAKE1_HEAD, SNAKE1_BODY, SNAKE2_HEAD, SNAKE2_BODY),
            shape=(296,),  # Fixed to match trained model
            dtype=np.float32
        )

        # Verify observation space size matches expected size
        expected_size = self.grid_size + n_additional_features
        if expected_size != 296:
            print(f"Warning: Current observation space size ({expected_size}) differs from trained model (296)")
            print(f"Grid size: {self.grid_size}, Additional features: {n_additional_features}")
            print("Feature breakdown:")
            if USE_DISTANCE_FEATURES:
                print("- Distance features: 7 (food, wall, opponent, rel_x, rel_y, body, powerup)")
            if USE_DANGER_FEATURES:
                print("- Danger features: 8 (4 immediate + 4 future)")
            if USE_FOOD_DIRECTION:
                print("- Direction features: 8 (4 food + 4 current)")
            print("- Other features: 1 (normalized length)")

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
        reward = 0.0

        if self.reward_approach == "A2":
            # Base reward for staying alive
            reward += REWARD_STEP_A2

            # Food rewards
            if snake1_ate_food:
                reward += REWARD_FOOD_A2

            # Powerup rewards
            if snake1_ate_powerup:
                reward += REWARD_POWERUP_A2

            # Death penalties
            if snake1_died:
                if info.get("snake1_death_reason") == "food_timeout":
                    reward += REWARD_TIMEOUT_A2
                else:
                    reward += REWARD_DEATH_A2
                    if info.get("snake1_death_reason") == "collision_opponent":
                        reward += REWARD_DEATH_BY_OPPONENT
            # Add hunger penalty when getting close to timeout
            elif self.snake1 and hasattr(self.snake1, 'steps_since_last_food') and self.snake1.steps_since_last_food > 45:  # 75% of timeout threshold
                reward += REWARD_HUNGER_A2

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
        if self.snake2 and hasattr(self.snake2, 'body'):
            for i, segment in enumerate(self.snake2.body):
                if 0 <= segment.y < self.grid_height and 0 <= segment.x < self.grid_width:
                    obs_grid[segment.y, segment.x] = SNAKE2_HEAD if i == 0 else SNAKE2_BODY

        # Add snake 1 (agent)
        if self.snake1 and hasattr(self.snake1, 'body'):
            for i, segment in enumerate(self.snake1.body):
                if 0 <= segment.y < self.grid_height and 0 <= segment.x < self.grid_width:
                    obs_grid[segment.y, segment.x] = SNAKE1_HEAD if i == 0 else SNAKE1_BODY

        # Flatten the grid
        flat_grid = obs_grid.flatten()

        # Calculate additional features
        additional_features = []

        if USE_DISTANCE_FEATURES:
            max_dist = self.grid_width + self.grid_height

            if self.snake1 and self.food:
                # Distance to food (normalized between 0 and 1)
                food_dist = self._get_manhattan_distance(self.snake1.body[0], self.food.position)
                norm_food_dist = food_dist / max_dist

                # Distance to nearest wall (normalized)
                wall_dist = self._get_nearest_wall_distance(self.snake1.body[0])
                norm_wall_dist = wall_dist / max_dist

                # Distance to opponent (normalized)
                if self.snake2:
                    opp_dist = self._get_manhattan_distance(self.snake1.body[0], self.snake2.body[0])
                    norm_opp_dist = opp_dist / max_dist
                else:
                    norm_opp_dist = 1.0  # Maximum normalized distance if no opponent

                additional_features.extend([norm_food_dist, norm_wall_dist, norm_opp_dist])

                # Add relative position features (normalized between -1 and 1)
                rel_x = (self.food.position.x - self.snake1.body[0].x) / self.grid_width
                rel_y = (self.food.position.y - self.snake1.body[0].y) / self.grid_height
                additional_features.extend([rel_x, rel_y])
            else:
                # Add placeholder values when food or snake1 doesn't exist
                additional_features.extend([1.0, 1.0, 1.0])  # max distances
                additional_features.extend([0.0, 0.0])  # neutral relative position

            # Add normalized distance to closest snake body part (excluding head)
            if self.snake1 and len(self.snake1.body) > 1:
                min_body_dist = min(self._get_manhattan_distance(self.snake1.body[0], segment)
                                  for segment in self.snake1.body[1:])
                norm_body_dist = min_body_dist / max_dist
            else:
                norm_body_dist = 1.0
            additional_features.append(norm_body_dist)

            # Add normalized distance to closest powerup
            if self.snake1 and self.powerup:
                powerup_dist = self._get_manhattan_distance(self.snake1.body[0], self.powerup.position)
                norm_powerup_dist = powerup_dist / max_dist
            else:
                norm_powerup_dist = 1.0
            additional_features.append(norm_powerup_dist)

        if USE_DANGER_FEATURES and self.snake1:
            # Check immediate danger in each direction
            dangers = self._get_danger_in_directions()
            additional_features.extend(dangers)

            # Add look-ahead danger (2 steps)
            future_dangers = self._get_future_danger()
            additional_features.extend(future_dangers)
        else:
            # Add placeholder values for danger features
            additional_features.extend([0.0] * 8)  # 4 immediate + 4 future dangers

        if USE_FOOD_DIRECTION and self.snake1 and self.food:
            # Relative direction to food
            food_dir = self._get_relative_direction(self.snake1.body[0], self.food.position)
            additional_features.extend(food_dir)

            # Add snake's current direction as one-hot
            current_dir = [0.0] * 4
            current_dir[Direction.DIR_TO_INDEX[self.snake1.direction]] = 1.0
            additional_features.extend(current_dir)
        else:
            # Add placeholder values for direction features
            additional_features.extend([0.0] * 8)  # 4 food direction + 4 current direction

        # Add snake length (normalized)
        if self.snake1:
            norm_length = len(self.snake1.body) / (self.grid_width * self.grid_height)
        else:
            norm_length = 0.0
        additional_features.append(norm_length)

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

    def _get_danger_in_directions(self, for_snake2=False):
        """Get danger in each direction for either snake."""
        snake = self.snake2 if for_snake2 else self.snake1
        other_snake = self.snake1 if for_snake2 else self.snake2

        if not snake or not hasattr(snake, 'body') or not snake.body:
            return [0.0] * 4

        head = snake.body[0]
        dangers = []

        for direction in Direction.INDEX_TO_DIR:  # Fixed: Use INDEX_TO_DIR to iterate over directions
            dx, dy = Direction.get_components(direction)
            next_pos = Point(head.x + dx, head.y + dy)
            # Check if next position would result in collision
            is_danger = (
                self.maze.is_wall(next_pos.x, next_pos.y) or
                next_pos in snake.body[1:] or
                (other_snake and hasattr(other_snake, 'body') and next_pos in other_snake.body)
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
            "snake1_score": self.snake1.score if self.snake1 else 0,
            "snake2_score": self.snake2.score if self.snake2 else 0,
            "steps": self._current_step,
            "snake1_len": len(self.snake1.body) if self.snake1 and hasattr(self.snake1, 'body') else 0,
            "snake2_len": len(self.snake2.body) if self.snake2 and hasattr(self.snake2, 'body') else 0,
            "snake1_death_reason": self.snake1_death_reason,
            "snake2_death_reason": self.snake2_death_reason,
        }

    def _place_item(self, item_class, **kwargs):
        occupied = set()
        if self.snake1 and hasattr(self.snake1, 'get_positions'):
            occupied |= self.snake1.get_positions()
        if self.snake2 and hasattr(self.snake2, 'get_positions'):
            occupied |= self.snake2.get_positions()
        if self.food:
            occupied.add(self.food.position)
        if self.powerup:
            occupied.add(self.powerup.position)

        pos = self.maze.get_random_empty_cell(occupied)
        return item_class(pos, **kwargs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0

        # Reset death reasons
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

        # Initialize snakes with random positions and directions
        snake1_start = Point(self.grid_width // 4, self.grid_height // 2)
        snake2_start = Point(3 * self.grid_width // 4, self.grid_height // 2)

        # Create snakes with initial directions facing each other
        self.snake1 = Snake(1, snake1_start, Direction.RIGHT, GREEN1, BLUE1)
        self.snake2 = Snake(2, snake2_start, Direction.LEFT, GREEN2, BLUE2)

        # Reset step counters and scores
        self.snake1.steps_since_last_food = 0
        self.snake2.steps_since_last_food = 0
        self.snake1.score = 0
        self.snake2.score = 0

        # Place food and powerup
        self.food = self._place_item(Food)

        # Place powerup with some probability
        if self.np_random.random() < POWERUP_SPAWN_CHANCE:
            self.powerup = self._place_item(PowerUp)
        else:
            self.powerup = None

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
                current_dir_index = Direction.DIR_TO_INDEX[self.snake2.direction]
                return current_dir_index
            except KeyError:
                return 0  # Default to UP if direction is invalid

        # Get valid actions for snake2
        valid_actions_mask = self._get_valid_actions_mask(for_snake2=True)
        valid_indices = [i for i, is_valid in enumerate(valid_actions_mask) if is_valid]

        # If no valid actions, return current direction (snake will die)
        if not valid_indices:
            try:
                return Direction.DIR_TO_INDEX[self.snake2.direction]
            except KeyError:
                return 0

        if self.opponent_policy_type == 'stay_still':
            # Try to maintain current direction if valid
            try:
                current_dir_index = Direction.DIR_TO_INDEX[self.snake2.direction]
                if current_dir_index in valid_indices:
                    return current_dir_index
                # If current direction is not valid, choose a random valid one
                return random.choice(valid_indices)
            except KeyError:
                return random.choice(valid_indices)

        elif self.opponent_policy_type == 'random':
            # Choose random valid action
            return random.choice(valid_indices)

        elif self.opponent_policy_type == 'basic_follow':
            # Very simple: move towards food if possible, else move randomly avoiding walls/self
            head = self.snake2.head
            target = self.food.position if self.food else None

            if target:
                # Calculate distances for each valid move
                best_action = -1
                min_dist = float('inf')

                for action in valid_indices:
                    direction = Direction.INDEX_TO_DIR[action]
                    dx, dy = Direction.get_components(direction)
                    next_pos = Point(head.x + dx, head.y + dy)
                    dist = abs(next_pos.x - target.x) + abs(next_pos.y - target.y)

                    if dist < min_dist:
                        min_dist = dist
                        best_action = action

                if best_action != -1:
                    return best_action

            # If no food or no path to food, choose random valid action
            return random.choice(valid_indices)
        else:
            # Default to random valid action
            return random.choice(valid_indices)

    def step(self, action):
        # Convert action to direction
        direction = Direction.INDEX_TO_DIR[action]

        # Initialize variables for reward calculation
        snake1_ate_food = False
        snake1_ate_powerup = False
        snake1_died = False
        snake2_died = False

        # Check if the move is valid
        if not self._is_valid_move(direction):
            self.snake1_death_reason = "invalid_move"
            snake1_died = True
        else:
            # Update snake1's direction and move
            self.snake1.direction = direction
            dx, dy = Direction.get_components(direction)
            new_head = Point(self.snake1.body[0].x + dx, self.snake1.body[0].y + dy)

            # Check for collisions with walls
            if self.maze.is_wall(new_head.x, new_head.y):
                self.snake1_death_reason = "collision_wall"
                snake1_died = True

            # Check for self-collision
            elif new_head in self.snake1.body[1:]:
                self.snake1_death_reason = "collision_self"
                snake1_died = True

            # Check for collision with opponent
            elif self.snake2 and new_head in self.snake2.body:
                self.snake1_death_reason = "collision_opponent"
                snake1_died = True

            else:
                # Move is safe, update snake position
                self.snake1.body.insert(0, new_head)

                # Check for food consumption
                if self.food and new_head == self.food.position:
                    snake1_ate_food = True
                    self.snake1.steps_since_last_food = 0
                    self.food = None
                    # Don't remove tail as snake grows
                else:
                    self.snake1.body.pop()
                    self.snake1.steps_since_last_food += 1

                # Check for powerup consumption
                if self.powerup and new_head == self.powerup.position:
                    snake1_ate_powerup = True
                    self.powerup = None

        # Check for food timeout for snake1
        if self.snake1 and not snake1_died and self.snake1.steps_since_last_food >= FOOD_TIMEOUT:
            self.snake1_death_reason = "food_timeout"
            snake1_died = True

        # Move opponent snake (snake2) if it exists and snake1 hasn't died
        if self.snake2 and not snake1_died:
            opponent_action = self._get_opponent_action()
            if opponent_action is not None:
                opp_direction = Direction.INDEX_TO_DIR[opponent_action]
                dx, dy = Direction.get_components(opp_direction)
                new_head = Point(self.snake2.body[0].x + dx, self.snake2.body[0].y + dy)

                # Check for opponent death conditions
                if self.maze.is_wall(new_head.x, new_head.y):
                    snake2_died = True
                    self.snake2_death_reason = "collision_wall"
                elif new_head in self.snake2.body[1:]:
                    snake2_died = True
                    self.snake2_death_reason = "collision_self"
                elif new_head in self.snake1.body:
                    snake2_died = True
                    self.snake2_death_reason = "collision_opponent"
                else:
                    self.snake2.direction = opp_direction
                    self.snake2.body.insert(0, new_head)

                    # Check if snake2 ate food
                    if self.food and new_head == self.food.position:
                        self.snake2.steps_since_last_food = 0
                        self.food = None
                    else:
                        self.snake2.body.pop()
                        self.snake2.steps_since_last_food += 1

                    # Check for food timeout for snake2
                    if self.snake2.steps_since_last_food >= FOOD_TIMEOUT:
                        snake2_died = True
                        self.snake2_death_reason = "food_timeout"

        # Spawn new food if needed and game isn't over
        if self.food is None and not (snake1_died or snake2_died):
            self.food = self._place_item(Food)

        # Spawn new powerup with small probability if game isn't over
        if self.powerup is None and random.random() < POWERUP_SPAWN_CHANCE and not (snake1_died or snake2_died):
            self.powerup = self._place_item(PowerUp)

        # Update step counter
        self._current_step += 1

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        info["snake1_death_reason"] = self.snake1_death_reason if snake1_died else ""
        info["snake2_death_reason"] = self.snake2_death_reason if snake2_died else ""

        # Calculate reward
        reward = self._get_reward(snake1_ate_food, snake1_ate_powerup, snake1_died, snake2_died, info)

        # Check if episode is done (either snake dies)
        done = snake1_died or snake2_died

        # Get valid actions mask for next step
        info["valid_actions"] = self._get_valid_actions_mask()

        return observation, reward, done, False, info

    def _is_valid_move(self, new_direction, for_snake2=False):
        """Check if the new direction is valid (not 180 degrees from current direction)."""
        snake = self.snake2 if for_snake2 else self.snake1
        if not snake:
            return False

        current = snake.direction
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        return new_direction != opposite[current]

    def _get_valid_actions_mask(self, for_snake2=False):
        """Return a boolean mask of valid actions."""
        snake = self.snake2 if for_snake2 else self.snake1
        other_snake = self.snake1 if for_snake2 else self.snake2

        if not snake:
            return [False] * 4

        valid_actions = []
        for direction in Direction.INDEX_TO_DIR:
            # Check if move is valid (not 180 degrees)
            is_valid = self._is_valid_move(direction, for_snake2=for_snake2)

            # Check if move leads to immediate death
            if is_valid:
                dx, dy = Direction.get_components(direction)
                next_pos = Point(snake.body[0].x + dx, snake.body[0].y + dy)
                is_valid = not (
                    self.maze.is_wall(next_pos.x, next_pos.y) or
                    next_pos in snake.body[1:] or
                    (other_snake and next_pos in other_snake.body)
                )

            valid_actions.append(is_valid)

        return valid_actions

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

        # Display Scores and Steps Since Food
        font = pygame.font.SysFont('arial', 20)
        # Snake 1 info
        if self.snake1:
            score_text1 = font.render(f"S1: {self.snake1.score}", True, GREEN1)
            steps_text1 = font.render(f"Steps: {self.snake1.steps_since_last_food}/60", True, GREEN1)
            canvas.blit(score_text1, (5, 5))
            canvas.blit(steps_text1, (5, 25))

        # Snake 2 info
        if self.snake2:
            score_text2 = font.render(f"S2: {self.snake2.score}", True, GREEN2)
            steps_text2 = font.render(f"Steps: {self.snake2.steps_since_last_food}/60", True, GREEN2)
            canvas.blit(score_text2, (self.window_width - 100, 5))
            canvas.blit(steps_text2, (self.window_width - 150, 25))

        # Display death reasons if any
        if self.snake1_death_reason:
            death_text1 = font.render(f"S1 died: {self.snake1_death_reason}", True, RED)
            canvas.blit(death_text1, (5, 45))
        if self.snake2_death_reason:
            death_text2 = font.render(f"S2 died: {self.snake2_death_reason}", True, RED)
            canvas.blit(death_text2, (self.window_width - 200, 45))

        if self.render_mode == "human":
            # Copy canvas to screen
            self.screen.blit(canvas, canvas.get_rect())
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        ) if self.render_mode == "rgb_array" else None

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

    def _get_future_danger(self):
        """Check for danger two steps ahead in each direction."""
        if not self.snake1:
            return [0.0] * 4

        dangers = []
        head = self.snake1.body[0]
        for direction in Direction.INDEX_TO_DIR:  # Fixed: Use INDEX_TO_DIR to iterate over directions
            # First step
            dx, dy = Direction.get_components(direction)
            next_pos = Point(head.x + dx, head.y + dy)

            # If first step is safe, check second step
            if not (self.maze.is_wall(next_pos.x, next_pos.y) or
                   next_pos in self.snake1.body[1:] or
                   (self.snake2 and next_pos in self.snake2.body)):
                # Check second step
                next_next_pos = Point(next_pos.x + dx, next_pos.y + dy)
                future_danger = float(
                    self.maze.is_wall(next_next_pos.x, next_next_pos.y) or
                    next_next_pos in self.snake1.body[1:] or
                    (self.snake2 and next_next_pos in self.snake2.body)
                )
                dangers.append(future_danger)
            else:
                dangers.append(1.0)  # First step is already dangerous

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
    # env = SnakeEnv(render_mode="human", maze_file="mazes/default.txt", reward_approach="A2", opponent_policy='basic_follow')
    env = SnakeEnv(render_mode="human", maze_file="mazes/maze1.txt", reward_approach="A2", opponent_policy='basic_follow') # Try a maze

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
