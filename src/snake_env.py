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
    """Snake environment with curriculum learning and two learning agents."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, maze_file="mazes/maze_natural.txt", reward_approach="A2", multi_agent=False, opponent_policy="basic_follow"):
        """Initialize the snake environment.

        Args:
            render_mode: How to render the environment
            maze_file: Path to the maze file
            reward_approach: Which reward structure to use
            multi_agent: Whether to use two learning agents
            opponent_policy: Policy for opponent in single-agent mode ('basic_follow', 'random', 'stay_still')
        """
        super().__init__()

        self.multi_agent = multi_agent
        self.opponent_policy = opponent_policy

        # Curriculum learning state
        self.curriculum_stage = 1 if CURRICULUM_ENABLED else 4
        self.stage_scores = []  # Track scores for stage advancement
        self.current_stage_config = CURRICULUM_STAGES[self.curriculum_stage]

        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.block_size = BLOCK_SIZE
        self.window_width = self.grid_width * self.block_size
        self.window_height = self.grid_height * self.block_size

        # Use maze file from curriculum if enabled, otherwise use provided
        self.maze = Maze(self.current_stage_config['maze_file'] if CURRICULUM_ENABLED else maze_file)
        self.reward_approach = reward_approach

        # Initialize game elements
        self.food = None
        self.powerup = None
        self.snake1 = None
        self.snake2 = None
        self.snake1_death_reason = ""
        self.snake2_death_reason = ""

        # Track episode statistics
        self.episode_steps = 0
        self.total_food_eaten = 0
        self.total_powerups_collected = 0
        self.snake2_total_food_eaten = 0
        self.snake2_total_powerups_collected = 0

        # Calculate observation space size
        self.grid_size = self.grid_width * self.grid_height
        n_additional_features = 0

        if USE_DISTANCE_FEATURES:
            n_additional_features += 7
        if USE_DANGER_FEATURES:
            n_additional_features += 8
        if USE_FOOD_DIRECTION:
            n_additional_features += 8
        if USE_LENGTH_FEATURE:
            n_additional_features += 1
        if USE_WALL_DISTANCE:
            n_additional_features += 4
        if USE_OPPONENT_FEATURES:
            n_additional_features += 4
        if USE_POWERUP_FEATURES:
            n_additional_features += 3

        # Define action and observation spaces (same for both agents)
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
            pygame.display.set_caption("Snake RL - Two Learning Agents")
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()

    def step(self, action):
        """
        Step function supporting both single and multi-agent modes.

        Args:
            action: Either a single action (int) for single-agent mode
                   or a dictionary {'snake1': action1, 'snake2': action2} for multi-agent mode
        """
        # Convert single action to action dict if needed
        action_dict = action if self.multi_agent else {'snake1': action, 'snake2': self._get_opponent_action()}

        # Process actions for both snakes
        snake1_action = action_dict['snake1']
        snake2_action = action_dict['snake2']

        # Initialize variables for reward calculation
        snake1_ate_food = False
        snake1_ate_powerup = False
        snake2_ate_food = False
        snake2_ate_powerup = False
        snake1_died = False
        snake2_died = False

        # Update episode counter
        self.episode_steps += 1

        # Process snake1's movement if it's not already dead
        if not self.snake1_death_reason:
            snake1_died = self._process_snake_movement(self.snake1, snake1_action, 1)
            if not snake1_died:
                # Check for food/powerup consumption for snake1
                snake1_ate_food, snake1_ate_powerup = self._check_consumables(self.snake1, 1)

        # Process snake2's movement if it's not already dead
        if not self.snake2_death_reason:
            snake2_died = self._process_snake_movement(self.snake2, snake2_action, 2)
            if not snake2_died:
                # Check for food/powerup consumption for snake2
                snake2_ate_food, snake2_ate_powerup = self._check_consumables(self.snake2, 2)

        # Check for food timeout for both snakes
        if self.snake1 and not self.snake1_death_reason and self.snake1.steps_since_last_food >= FOOD_TIMEOUT:
            self.snake1_death_reason = "food_timeout"
            snake1_died = True

        if self.snake2 and not self.snake2_death_reason and self.snake2.steps_since_last_food >= FOOD_TIMEOUT:
            self.snake2_death_reason = "food_timeout"
            snake2_died = True

        # Spawn new food if needed and at least one snake is alive
        if self.food is None and (not self.snake1_death_reason or not self.snake2_death_reason):
            max_distance = self.current_stage_config.get('food_max_distance') if CURRICULUM_ENABLED else None
            self.food = self._place_item(Food, max_distance=max_distance)

        # Spawn new powerup if enabled and at least one snake is alive
        if (self.powerup is None and
            self.current_stage_config['powerups_enabled'] and
            random.random() < POWERUP_SPAWN_CHANCE and
            (not self.snake1_death_reason or not self.snake2_death_reason)):
            self.powerup = self._place_item(PowerUp)

        # Get observations and info for both snakes
        snake1_obs = self._get_obs(1)
        snake2_obs = self._get_obs(2)

        info = {
            'snake1': {
                'death_reason': self.snake1_death_reason if snake1_died else "",
                'score': self.snake1.score if self.snake1 else 0,
                'valid_actions': self._get_valid_actions_mask(1)
            },
            'snake2': {
                'death_reason': self.snake2_death_reason if snake2_died else "",
                'score': self.snake2.score if self.snake2 else 0,
                'valid_actions': self._get_valid_actions_mask(2)
            }
        }

        # Calculate rewards for both snakes
        snake1_reward = self._get_reward(snake1_ate_food, snake1_ate_powerup, snake1_died, snake2_died, info['snake1'], 1)
        snake2_reward = self._get_reward(snake2_ate_food, snake2_ate_powerup, snake2_died, snake1_died, info['snake2'], 2)

        # Episode ends only when snake1 (the primary learning agent) dies
        done = bool(self.snake1_death_reason)

        # Update curriculum learning scores if episode is done
        if done:
            self.stage_scores.append(self.snake1.score)

        if self.render_mode == "human":
            self._render_frame()

        # Return appropriate format based on mode
        if self.multi_agent:
            return {
                'snake1': (snake1_obs, snake1_reward, done, False, info['snake1']),
                'snake2': (snake2_obs, snake2_reward, bool(self.snake2_death_reason), False, info['snake2'])
            }
        else:
            return snake1_obs, snake1_reward, done, False, info['snake1']

    def _get_obs(self, snake_id=1):
        """Get observation for specified snake."""
        obs_grid = np.full((self.grid_height, self.grid_width), EMPTY, dtype=np.float32)
        self._add_basic_elements_to_grid(obs_grid)

        # Get the snake for which we're generating the observation
        snake = self.snake1 if snake_id == 1 else self.snake2
        opponent = self.snake2 if snake_id == 1 else self.snake1

        # Flatten the grid
        flat_grid = obs_grid.flatten()
        additional_features = []

        if snake and hasattr(snake, 'body'):
            head = snake.body[0]

            if USE_DISTANCE_FEATURES:
                additional_features.extend(self._get_distance_features(head, snake_id))

            if USE_DANGER_FEATURES:
                additional_features.extend(self._get_danger_features(snake_id))

            if USE_FOOD_DIRECTION:
                additional_features.extend(self._get_food_direction_features(head))

            if USE_LENGTH_FEATURE:
                additional_features.append(len(snake.body) / (self.grid_width * self.grid_height))

            if USE_WALL_DISTANCE:
                additional_features.extend(self._get_wall_distance_features(head))

            if USE_OPPONENT_FEATURES and opponent and not opponent.death_reason:
                additional_features.extend(self._get_opponent_features(head, opponent))
            else:
                additional_features.extend([0.0] * 4)

            if USE_POWERUP_FEATURES:
                additional_features.extend(self._get_powerup_features(head))
        else:
            # Add zero features if snake doesn't exist
            n_features = sum([
                7 if USE_DISTANCE_FEATURES else 0,
                8 if USE_DANGER_FEATURES else 0,
                8 if USE_FOOD_DIRECTION else 0,
                1 if USE_LENGTH_FEATURE else 0,
                4 if USE_WALL_DISTANCE else 0,
                4 if USE_OPPONENT_FEATURES else 0,
                3 if USE_POWERUP_FEATURES else 0
            ])
            additional_features = [0.0] * n_features

        return np.concatenate([flat_grid, np.array(additional_features, dtype=np.float32)])

    def _add_basic_elements_to_grid(self, grid):
        """Add basic elements (walls, snakes, food, powerups) to the observation grid."""
        # Add walls
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if self.maze.is_wall(x, y):
                    grid[y, x] = WALL

        def is_valid_position(pos):
            """Check if a position is within grid bounds."""
            return (0 <= pos.x < self.grid_width and
                   0 <= pos.y < self.grid_height)

        # Add snake1 if it exists
        if self.snake1 and hasattr(self.snake1, 'body'):
            # Add body
            for segment in self.snake1.body[1:]:
                if is_valid_position(segment):
                    grid[segment.y, segment.x] = SNAKE1_BODY
            # Add head
            if self.snake1.body:
                head = self.snake1.body[0]
                if is_valid_position(head):
                    grid[head.y, head.x] = SNAKE1_HEAD

        # Add snake2 if it exists
        if self.snake2 and hasattr(self.snake2, 'body'):
            # Add body
            for segment in self.snake2.body[1:]:
                if is_valid_position(segment):
                    grid[segment.y, segment.x] = SNAKE2_BODY
            # Add head
            if self.snake2.body:
                head = self.snake2.body[0]
                if is_valid_position(head):
                    grid[head.y, head.x] = SNAKE2_HEAD

        # Add food
        if self.food and hasattr(self.food, 'position'):
            pos = self.food.position
            if is_valid_position(pos):
                grid[pos.y, pos.x] = FOOD_ITEM

        # Add powerup
        if self.powerup and hasattr(self.powerup, 'position'):
            pos = self.powerup.position
            if is_valid_position(pos):
                grid[pos.y, pos.x] = POWERUP_ITEM

    def _get_reward(self, ate_food, ate_powerup, died, opponent_died, info, snake_id=1):
        """Calculate reward for specified snake."""
        reward = 0
        snake = self.snake1 if snake_id == 1 else self.snake2
        opponent = self.snake2 if snake_id == 1 else self.snake1

        if self.reward_approach == "A2":
            # Base rewards
            reward += REWARD_STEP_A2
            if ate_food:
                reward += REWARD_FOOD_A2
            if ate_powerup:
                reward += REWARD_POWERUP_A2
            if died:
                reward += REWARD_DEATH_A2
            if died and info['death_reason'] == "food_timeout":
                reward += REWARD_TIMEOUT_A2

            # Progressive rewards
            if snake and hasattr(snake, 'body'):
                head = snake.body[0]
                if self.food:
                    new_dist = self._get_manhattan_distance(head, self.food.position)
                    if hasattr(snake, 'prev_food_distance'):
                        if new_dist < snake.prev_food_distance:
                            reward += REWARD_CLOSER_TO_FOOD
                        elif new_dist > snake.prev_food_distance:
                            reward += REWARD_FURTHER_FROM_FOOD
                    snake.prev_food_distance = new_dist

                # Wall distance reward
                if USE_WALL_DISTANCE:
                    min_wall_dist = min(
                        head.x,
                        head.y,
                        self.grid_width - 1 - head.x,
                        self.grid_height - 1 - head.y
                    )
                    reward += REWARD_WALL_DISTANCE * min_wall_dist

                # Opponent distance reward
                if opponent and not opponent.death_reason:
                    opp_dist = self._get_manhattan_distance(head, opponent.body[0])
                    reward += REWARD_OPPONENT_DISTANCE * min(opp_dist, 5)

                # Hunger penalty
                if snake.steps_since_last_food > 45:
                    reward += REWARD_HUNGER_A2

        return reward

    def reset(self, seed=None, options=None):
        """Reset environment supporting both single and multi-agent modes."""
        super().reset(seed=seed)

        # Reset environment state
        self.snake1_death_reason = ""
        self.snake2_death_reason = ""
        self.episode_steps = 0
        self.food = None
        self.powerup = None

        # Initialize snakes with proper parameters
        start_pos1 = self._get_random_empty_position()
        start_dir1 = random.choice(list(Direction.INDEX_TO_DIR))
        self.snake1 = Snake(1, start_pos1, start_dir1, GREEN1, BLUE1)

        # Ensure second snake starts in a different position
        while True:
            start_pos2 = self._get_random_empty_position()
            if start_pos2 != start_pos1:
                break
        start_dir2 = random.choice(list(Direction.INDEX_TO_DIR))
        self.snake2 = Snake(2, start_pos2, start_dir2, GREEN2, BLUE2)

        # Place initial food
        max_distance = self.current_stage_config.get('food_max_distance') if CURRICULUM_ENABLED else None
        self.food = self._place_item(Food, max_distance=max_distance)

        # Get initial observations
        snake1_obs = self._get_obs(1)
        snake2_obs = self._get_obs(2)

        if self.render_mode == "human":
            self._render_frame()

        # Return appropriate format based on mode
        if self.multi_agent:
            return {
                'snake1': (snake1_obs, {}),
                'snake2': (snake2_obs, {})
            }
        else:
            return snake1_obs, {}

    def _check_curriculum_advancement(self):
        """Check if we should advance to the next curriculum stage."""
        if not CURRICULUM_ENABLED or self.curriculum_stage >= 4:
            return False

        stage_config = CURRICULUM_STAGES[self.curriculum_stage]
        min_score = stage_config['min_score_advance']

        # Keep only last 10 episodes for evaluation
        self.stage_scores = self.stage_scores[-10:]

        # Check if ready to advance
        if len(self.stage_scores) >= 5:  # Need at least 5 episodes to evaluate
            avg_score = sum(self.stage_scores) / len(self.stage_scores)
            if avg_score >= min_score:
                self.curriculum_stage += 1
                self.stage_scores = []  # Reset scores for new stage
                self.current_stage_config = CURRICULUM_STAGES[self.curriculum_stage]
                return True
        return False

    def _get_distance_features(self, head, snake_id):
        """Calculate distance-based features."""
        max_dist = self.grid_width + self.grid_height
        features = []

        # Distance to food
        if self.food:
            food_dist = self._get_manhattan_distance(head, self.food.position)
            norm_food_dist = food_dist / max_dist
            # Relative position to food
            rel_x = (self.food.position.x - head.x) / self.grid_width
            rel_y = (self.food.position.y - head.y) / self.grid_height
        else:
            norm_food_dist = 1.0
            rel_x = rel_y = 0.0

        # Distance to nearest wall
        wall_dist = self._get_nearest_wall_distance(head)
        norm_wall_dist = wall_dist / max_dist

        # Distance to opponent
        if snake_id == 1 and self.snake2:
            opp_dist = self._get_manhattan_distance(head, self.snake2.body[0])
            norm_opp_dist = opp_dist / max_dist
        else:
            norm_opp_dist = 1.0

        # Distance to own body
        if len(self.snake1.body) > 1:
            body_dist = min(self._get_manhattan_distance(head, segment)
                          for segment in self.snake1.body[1:])
            norm_body_dist = body_dist / max_dist
        else:
            norm_body_dist = 1.0

        features.extend([norm_food_dist, norm_wall_dist, norm_opp_dist, rel_x, rel_y, norm_body_dist])

        # Distance to powerup
        if self.powerup:
            powerup_dist = self._get_manhattan_distance(head, self.powerup.position)
            norm_powerup_dist = powerup_dist / max_dist
        else:
            norm_powerup_dist = 1.0
        features.append(norm_powerup_dist)

        return features

    def _get_danger_features(self, snake_id):
        """Get danger detection features."""
        if not self.snake1 and snake_id == 1 or not self.snake2 and snake_id == 2:
            return [0.0] * 8

        # Immediate danger in each direction
        immediate_dangers = self._get_danger_in_directions(snake_id)

        # Future danger (2 steps ahead)
        future_dangers = self._get_future_danger(snake_id)

        return immediate_dangers + future_dangers

    def _get_future_danger(self, snake_id=1):
        """Calculate potential dangers two steps ahead."""
        snake = self.snake1 if snake_id == 1 else self.snake2
        other_snake = self.snake2 if snake_id == 1 else self.snake1

        if not snake or not hasattr(snake, 'body') or not snake.body:
            return [0.0] * 4

        head = snake.body[0]
        future_dangers = []

        for direction in Direction.INDEX_TO_DIR:
            # First step
            dx, dy = Direction.get_components(direction)
            next_pos = Point(head.x + dx, head.y + dy)

            # Check if first step is within bounds
            if (next_pos.x < 0 or next_pos.x >= self.grid_width or
                next_pos.y < 0 or next_pos.y >= self.grid_height):
                future_dangers.append(1.0)
                continue

            # Check if first step hits a wall or snake
            if (self.maze.is_wall(next_pos.x, next_pos.y) or
                (snake and hasattr(snake, 'body') and next_pos in snake.body[1:]) or
                (other_snake and hasattr(other_snake, 'body') and next_pos in other_snake.body)):
                future_dangers.append(1.0)
                continue

            # Second step - check all possible directions from next_pos
            second_step_danger = False
            for second_direction in Direction.INDEX_TO_DIR:
                dx2, dy2 = Direction.get_components(second_direction)
                future_pos = Point(next_pos.x + dx2, next_pos.y + dy2)

                # Check if second step is within bounds
                if (future_pos.x < 0 or future_pos.x >= self.grid_width or
                    future_pos.y < 0 or future_pos.y >= self.grid_height):
                    continue

                # Check if at least one safe path exists
                if (not self.maze.is_wall(future_pos.x, future_pos.y) and
                    (not snake or not hasattr(snake, 'body') or future_pos not in snake.body[1:]) and
                    (not other_snake or not hasattr(other_snake, 'body') or future_pos not in other_snake.body)):
                    second_step_danger = False
                    break
                second_step_danger = True

            future_dangers.append(1.0 if second_step_danger else 0.0)

        return future_dangers

    def _get_food_direction_features(self, head):
        """Get food direction and current direction features."""
        features = []

        # Food direction one-hot
        if self.food:
            food_dir = self._get_relative_direction(head, self.food.position)
        else:
            food_dir = [0.0] * 4
        features.extend(food_dir)

        # Current direction one-hot
        current_dir = [0.0] * 4
        if self.snake1 and hasattr(self.snake1, 'direction'):
            current_dir[Direction.DIR_TO_INDEX[self.snake1.direction]] = 1.0
        elif self.snake2 and hasattr(self.snake2, 'direction'):
            current_dir[Direction.DIR_TO_INDEX[self.snake2.direction]] = 1.0
        features.extend(current_dir)

        return features

    def _get_wall_distance_features(self, head):
        """Calculate distance to walls in each direction."""
        distances = []
        for direction in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # up, down, left, right
            dist = 0
            x, y = head.x, head.y
            while 0 <= x < self.grid_width and 0 <= y < self.grid_height and not self.maze.is_wall(x, y):
                dist += 1
                x += direction[0]
                y += direction[1]
            distances.append(dist / max(self.grid_width, self.grid_height))
        return distances

    def _get_opponent_features(self, head, opponent):
        """Calculate opponent-related features."""
        if not opponent or not hasattr(opponent, 'body'):
            return [0.0] * 4

        opp_head = opponent.body[0]
        rel_x = (opp_head.x - head.x) / self.grid_width
        rel_y = (opp_head.y - head.y) / self.grid_height

        # Opponent movement direction (normalized)
        opp_dx = 0
        opp_dy = 0
        if len(opponent.body) > 1:
            prev_head = opponent.body[1]
            opp_dx = (opp_head.x - prev_head.x) / self.grid_width
            opp_dy = (opp_head.y - prev_head.y) / self.grid_height

        return [rel_x, rel_y, opp_dx, opp_dy]

    def _get_powerup_features(self, head):
        """Calculate powerup-related features."""
        if not self.powerup:
            return [0.0] * 3

        dist = self._get_manhattan_distance(head, self.powerup.position)
        norm_dist = dist / (self.grid_width + self.grid_height)

        # Powerup type one-hot (currently only one type)
        type_feature = 1.0 if self.powerup.type == 'extra_points' else 0.0

        return [norm_dist, type_feature, float(self.powerup.points) / 10.0]

    def _get_manhattan_distance(self, point1, point2):
        return abs(point1.x - point2.x) + abs(point1.y - point2.y)

    def _get_nearest_wall_distance(self, point):
        distances = []
        for wall in self.maze.barriers:
            dist = self._get_manhattan_distance(point, Point(wall[0], wall[1]))
            distances.append(dist)
        return min(distances) if distances else self.grid_width

    def _get_danger_in_directions(self, snake_id):
        """Get danger in each direction for specified snake."""
        snake = self.snake1 if snake_id == 1 else self.snake2
        other_snake = self.snake1 if snake_id == 1 else self.snake2

        if not snake or not hasattr(snake, 'body') or not snake.body:
            return [0.0] * 4

        head = snake.body[0]
        dangers = []

        for direction in Direction.INDEX_TO_DIR:
            dx, dy = Direction.get_components(direction)
            next_pos = Point(head.x + dx, head.y + dy)

            # Initialize danger as False
            is_danger = False

            # Check wall collision
            if next_pos.x < 0 or next_pos.x >= self.grid_width or next_pos.y < 0 or next_pos.y >= self.grid_height:
                is_danger = True
            else:
                # Check wall collision
                is_danger = (
                    self.maze.is_wall(next_pos.x, next_pos.y) or
                    # Check self collision
                    (snake and hasattr(snake, 'body') and next_pos in snake.body[1:]) or
                    # Check opponent collision
                    (other_snake and hasattr(other_snake, 'body') and next_pos in other_snake.body)
                )

            dangers.append(1.0 if is_danger else 0.0)

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
            "steps": self.episode_steps,
            "snake1_len": len(self.snake1.body) if self.snake1 and hasattr(self.snake1, 'body') else 0,
            "snake2_len": len(self.snake2.body) if self.snake2 and hasattr(self.snake2, 'body') else 0,
            "snake1_death_reason": self.snake1_death_reason,
            "snake2_death_reason": self.snake2_death_reason,
        }

    def _is_path_relatively_clear(self, start, end):
        """Check if there's a relatively clear path between two points."""
        # Get the direct path coordinates
        x0, y0 = start.x, start.y
        x1, y1 = end.x, end.y
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        # Check points along the path
        path_points = []
        obstacle_count = 0
        for _ in range(min(n, 10)):  # Limit check to 10 steps
            if self.maze.is_wall(x, y):
                obstacle_count += 1
            if obstacle_count > 0:  # Even stricter - no obstacles allowed in early training
                return False

            # Check for snake body parts blocking the path
            if self.snake1 and hasattr(self.snake1, 'body'):
                if Point(x, y) in self.snake1.body[1:]:
                    return False
            if self.snake2 and hasattr(self.snake2, 'body'):
                if Point(x, y) in self.snake2.body:
                    return False

            path_points.append(Point(x, y))
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        # Early training: ensure path is completely clear
        if self.total_food_eaten < 10:
            return obstacle_count == 0
        # Later training: allow at most one obstacle
        return obstacle_count <= 1

    def _place_item(self, item_class, max_distance=None, **kwargs):
        """Place an item in a valid position on the grid."""
        occupied = set()
        if self.snake1 and hasattr(self.snake1, 'get_positions'):
            occupied |= self.snake1.get_positions()
        if self.snake2 and hasattr(self.snake2, 'get_positions'):
            occupied |= self.snake2.get_positions()
        if self.food:
            occupied.add(self.food.position)
        if self.powerup:
            occupied.add(self.powerup.position)

        # Early training adjustment: if no max_distance specified and it's food, use a dynamic distance
        if max_distance is None and item_class == Food:
            if self.total_food_eaten < 3:
                max_distance = 2  # Start extremely close
            elif self.total_food_eaten < 5:
                max_distance = 3  # Very close
            elif self.total_food_eaten < 10:
                max_distance = 4  # Still quite close
            elif self.total_food_eaten < 15:
                max_distance = 6  # Gradually increase

        # Get all valid positions
        valid_positions = []
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                pos = Point(x, y)
                if not self.maze.is_wall(x, y) and pos not in occupied:
                    if max_distance is not None and self.snake1:
                        dist = self._get_manhattan_distance(self.snake1.body[0], pos)
                        if dist <= max_distance:
                            # Add position with priority based on distance and clear path
                            path_clear = self._is_path_relatively_clear(self.snake1.body[0], pos)
                            # In early training, only consider positions with clear paths
                            if self.total_food_eaten < 10 and not path_clear:
                                continue
                            priority = 1.0 if path_clear else 0.5
                            valid_positions.append((pos, dist, priority))
                    else:
                        valid_positions.append((pos, 0, 1.0))

        # If no valid positions found within max_distance, gradually increase the distance
        original_max_distance = max_distance
        while not valid_positions and max_distance is not None:
            max_distance += 1
            if max_distance > self.grid_width + self.grid_height or max_distance > 2 * original_max_distance:
                break
            for x in range(self.grid_width):
                for y in range(self.grid_height):
                    pos = Point(x, y)
                    if not self.maze.is_wall(x, y) and pos not in occupied:
                        dist = self._get_manhattan_distance(self.snake1.body[0], pos)
                        if dist <= max_distance:
                            path_clear = self._is_path_relatively_clear(self.snake1.body[0], pos)
                            if self.total_food_eaten < 10 and not path_clear:
                                continue
                            priority = 1.0 if path_clear else 0.5
                            valid_positions.append((pos, dist, priority))

        if not valid_positions:
            # If still no valid positions, try again without path clarity requirement
            for x in range(self.grid_width):
                for y in range(self.grid_height):
                    pos = Point(x, y)
                    if not self.maze.is_wall(x, y) and pos not in occupied:
                        valid_positions.append((pos, 0, 0.5))

        if not valid_positions:
            return None  # No valid positions available

        # Sort positions by distance and priority
        if item_class == Food:
            valid_positions.sort(key=lambda x: (x[1], -x[2]))  # Sort by distance first, then by priority
            if self.total_food_eaten < 15:
                # In early training, always choose from the closest positions with highest priority
                best_positions = [p for p in valid_positions[:4] if p[2] == 1.0]
                if not best_positions:
                    best_positions = valid_positions[:2]
                pos = random.choice([p[0] for p in best_positions])
            else:
                # Later in training, use a wider selection but still prefer clear paths
                best_positions = [p for p in valid_positions[:8] if p[2] == 1.0]
                if not best_positions:
                    best_positions = valid_positions[:4]
                pos = random.choice([p[0] for p in best_positions])
        else:
            pos = random.choice([p[0] for p in valid_positions])

        return item_class(pos)

    def _process_snake_movement(self, snake, action, snake_id):
        """Process the movement of a specified snake."""
        direction = Direction.INDEX_TO_DIR[action]
        dx, dy = Direction.get_components(direction)
        new_head = Point(snake.body[0].x + dx, snake.body[0].y + dy)

        # Check for collisions
        if self.maze.is_wall(new_head.x, new_head.y):
            snake.death_reason = "collision_wall"
            return True
        elif new_head in snake.body[1:]:
            snake.death_reason = "collision_self"
            return True
        elif snake_id == 1 and self.snake2 and not self.snake2.death_reason and new_head in self.snake2.body:
            snake.death_reason = "collision_opponent"
            return True
        else:
            # Move is safe, update snake position
            snake.body.insert(0, new_head)

            # Check for food consumption
            if self.food and new_head == self.food.position:
                snake.steps_since_last_food = 0
                snake.score += 1
                self.food = None
            else:
                snake.body.pop()
                snake.steps_since_last_food += 1

            return False

    def _check_consumables(self, snake, snake_id):
        """Check for food and powerup consumption for a specified snake."""
        ate_food = False
        ate_powerup = False
        if self.food and self.food.position == snake.body[0]:
            ate_food = True
            snake.steps_since_last_food = 0
            snake.score += 1
            self.food = None
        elif self.powerup and self.powerup.position == snake.body[0]:
            ate_powerup = True
            snake.steps_since_last_food = 0
            snake.score += 1
            self.powerup = None
        return ate_food, ate_powerup

    def _is_valid_move(self, new_direction, snake_id):
        """Check if a move is valid (not 180 degrees from current direction)."""
        snake = self.snake1 if snake_id == 1 else self.snake2

        if not snake or not hasattr(snake, 'direction'):
            return True  # Any move is valid if snake doesn't exist or has no direction

        current_dir = snake.direction

        # Opposite direction pairs
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }

        # Move is invalid if it's opposite to current direction
        return new_direction != opposites.get(current_dir, current_dir)

    def _get_valid_actions_mask(self, snake_id):
        """Return a boolean mask of valid actions for specified snake."""
        snake = self.snake1 if snake_id == 1 else self.snake2
        other_snake = self.snake1 if snake_id == 1 else self.snake2

        if not snake:
            return [False] * 4

        valid_actions = []
        for direction in Direction.INDEX_TO_DIR:
            # Check if move is valid (not 180 degrees)
            is_valid = self._is_valid_move(direction, snake_id)

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
            pygame.display.set_caption("Snake RL - Two Learning Agents")
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

    def _get_random_empty_position(self):
        """Get a random empty position within the grid bounds."""
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(1, self.grid_width - 2)  # Leave 1-cell border
            y = random.randint(1, self.grid_height - 2)  # Leave 1-cell border
            pos = Point(x, y)

            # Check if position is empty
            if (not self.maze.is_wall(x, y) and
                (not self.snake1 or not hasattr(self.snake1, 'body') or pos not in self.snake1.body) and
                (not self.snake2 or not hasattr(self.snake2, 'body') or pos not in self.snake2.body) and
                (not self.food or not hasattr(self.food, 'position') or pos != self.food.position) and
                (not self.powerup or not hasattr(self.powerup, 'position') or pos != self.powerup.position)):
                return pos

        # If no empty position found, use a safe default
        return Point(self.grid_width // 2, self.grid_height // 2)

    def update_maze(self, new_maze_file):
        """Update the current maze during training"""
        print(f"Loading maze from {new_maze_file}")

        # Don't reload maze if it's the same, but still reset game state
        if self.maze and self.maze.filepath == new_maze_file and self.maze.is_loaded:
            print(f"Maze {new_maze_file} is already loaded, resetting game state")
            try:
                # Reset game state
                self.episode_steps = 0
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
                self.episode_steps = 0
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

    def _get_opponent_action(self):
        """Get action for non-learning opponent in single-agent mode."""
        if not self.snake2 or self.snake2_death_reason:
            return 0  # Default action if snake is dead

        # Get valid actions
        valid_actions = self._get_valid_actions_mask(2)
        valid_indices = [i for i, valid in enumerate(valid_actions) if valid]

        if not valid_indices:
            return 0  # Default action if no valid moves

        if self.opponent_policy == "stay_still":
            # Try to maintain current direction, or choose random valid direction if current is invalid
            if hasattr(self.snake2, 'direction'):
                current_action = Direction.DIR_TO_INDEX[self.snake2.direction]
                if current_action in valid_indices:
                    return current_action

        elif self.opponent_policy == "basic_follow":
            # Move towards food if possible
            if self.food and hasattr(self.snake2, 'body'):
                head = self.snake2.body[0]
                food_dir = self._get_relative_direction(head, self.food.position)
                food_action = food_dir.index(1.0) if 1.0 in food_dir else None
                if food_action in valid_indices:
                    return food_action

        # For "random" policy or if other policies' preferred actions aren't valid
        return random.choice(valid_indices)

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
    env = SnakeEnv(render_mode="human", maze_file="mazes/maze1.txt", reward_approach="A2") # Try a maze

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
        action_dict = {'snake1': env.action_space.sample(), 'snake2': env.action_space.sample()}  # Take random actions
        obs, reward, terminated, truncated, info = env.step(action_dict)
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
