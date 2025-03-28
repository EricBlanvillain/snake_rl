# visualize.py
import pygame
import os
import time
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
import sys
import json
import random

from snake_env import SnakeEnv, Direction, Point
from constants import *
from menu import Menu
from button import Button
from leaderboard import Leaderboard

# Constants for visualization
CELL_SIZE = 30
GRID_LINE_WIDTH = 1
FPS = 15

class RetroVisualizer:
    def __init__(self, env, model1=None, model2=None):
        self.env = env
        self.model1 = model1
        self.model2 = model2
        self.paused = False
        self.screen = None
        self.clock = pygame.time.Clock()
        self.font = None
        self.episode = 0
        self.steps = 0
        self.menu = None
        self.pause_buttons = []
        self.death_reason = ""
        self.death_display_time = None
        self.show_leaderboard = False
        self.leaderboard = Leaderboard()
        self._score_recorded = False
        self.game_over = False

        # Get maze name safely
        if hasattr(self.env.maze, 'maze_file') and self.env.maze.maze_file:
            self.current_maze_name = os.path.splitext(os.path.basename(self.env.maze.maze_file))[0]
        else:
            self.current_maze_name = "default_maze"

        self.death_messages = {
            "collision_wall": "Hit a wall!",
            "collision_self": "Self collision!",
            "collision_opponent": "Hit opponent snake!",
            "food_timeout": "Starved!",
            "collision_opponent_head": "Head-on collision!",
            "collision_opponent_body": "Hit opponent's body!"
        }

        # Initialize Pygame and font system
        pygame.init()
        pygame.font.init()  # Explicitly initialize font system
        pygame.display.set_caption("Snake RL - Retro Punk Edition")

        # Calculate window dimensions based on grid size
        self.screen_width = self.env.grid_width * CELL_SIZE
        self.screen_height = self.env.grid_height * CELL_SIZE + 60  # Extra space for score

        # Initialize menu
        self.menu = Menu(self.screen_width, self.screen_height)

        try:
            # Try to load custom font
            font_path = os.path.join(os.path.dirname(__file__), "assets", "Streamster.ttf")
            if os.path.exists(font_path):
                self.font = pygame.font.Font(font_path, 24)
            else:
                print("Custom font not found, using default")
                self.font = pygame.font.SysFont('arial', 24)
        except Exception as e:
            print(f"Error loading font: {e}")
            self.font = pygame.font.SysFont('arial', 24)

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        # Initialize pause menu buttons
        button_width = 200
        button_height = 50
        button_x = (self.screen_width - button_width) // 2

        self.pause_buttons = [
            Button(button_x, self.screen_height//2 - 60, button_width, button_height, "Resume", (0, 255, 100)),
            Button(button_x, self.screen_height//2, button_width, button_height, "Select Maze", (0, 150, 255)),
            Button(button_x, self.screen_height//2 + 60, button_width, button_height, "Quit", (255, 50, 50))
        ]

    def draw_grid(self):
        """Draw the game grid with a retro effect."""
        for x in range(0, self.screen_width, CELL_SIZE):
            pygame.draw.line(self.screen, COLORS['grid'], (x, 0), (x, self.screen_height - 60), GRID_LINE_WIDTH)
        for y in range(0, self.screen_height - 60, CELL_SIZE):
            pygame.draw.line(self.screen, COLORS['grid'], (0, y), (self.screen_width, y), GRID_LINE_WIDTH)

    def draw_cell(self, x, y, color, glow=False):
        """Draw a cell with optional neon glow effect."""
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

        if glow:
            # Draw glow effect
            surf = pygame.Surface((CELL_SIZE + 10, CELL_SIZE + 10), pygame.SRCALPHA)
            pygame.draw.rect(surf, (*color[:3], 100), (5, 5, CELL_SIZE, CELL_SIZE))
            self.screen.blit(surf, (x * CELL_SIZE - 5, y * CELL_SIZE - 5))

        pygame.draw.rect(self.screen, color, rect)
        # Add inner rectangle for depth
        pygame.draw.rect(self.screen, (color[0]//2, color[1]//2, color[2]//2),
                        rect.inflate(-8, -8))

    def draw_snake(self, snake, color):
        """Draw a snake with neon effect."""
        for i, segment in enumerate(snake.body):
            self.draw_cell(segment.x, segment.y, color, glow=(i == 0))  # Glow effect on head

    def draw_score(self):
        """Draw score, episode info, and death reason with retro style."""
        # Safely get snake scores
        snake1_score = self.env.snake1.score if self.env.snake1 else 0
        snake2_score = self.env.snake2.score if self.env.snake2 else 0

        # Create shaded background for text
        text_bg = pygame.Surface((self.screen_width, 60))
        text_bg.fill(COLORS['background'])
        text_bg.set_alpha(200)
        self.screen.blit(text_bg, (0, self.screen_height - 60))

        # Render scores with larger font and glow effect
        score_font = pygame.font.SysFont('arial', 32)  # Larger font for scores
        info_font = pygame.font.SysFont('arial', 24)   # Regular font for other info

<<<<<<< HEAD
        # Get current maze name and create subtle text
        maze_name = os.path.basename(self.env.maze.filepath) if hasattr(self.env.maze, 'filepath') else "Unknown Maze"
        maze_font = pygame.font.SysFont('arial', 18)  # Smaller font for subtlety
        maze_text = maze_font.render(maze_name, True, (100, 100, 120))  # Subtle gray color
        maze_text.set_alpha(150)  # Make it semi-transparent

        # Position all elements
        self.screen.blit(score_surf, (10, self.screen_height - 55))
        self.screen.blit(episode_surf, (self.screen_width//2 - episode_surf.get_width()//2, self.screen_height - 55))
        self.screen.blit(steps_surf, (self.screen_width - steps_surf.get_width() - 10, self.screen_height - 55))
        self.screen.blit(maze_text, (self.screen_width//2 - maze_text.get_width()//2, self.screen_height - 25))  # Position at bottom
=======
        # Snake 1 score with glow
        score1_text = f"P1: {snake1_score}"
        score1_surf = score_font.render(score1_text, True, COLORS['snake1'])
        # Add glow effect
        glow_surf = pygame.Surface((score1_surf.get_width() + 10, score1_surf.get_height() + 10), pygame.SRCALPHA)
        glow_surf.blit(score1_surf, (5, 5))
        glow_surf.set_alpha(100)
        self.screen.blit(glow_surf, (5, self.screen_height - 55))
        self.screen.blit(score1_surf, (10, self.screen_height - 50))

        # Snake 2 score with glow
        score2_text = f"P2: {snake2_score}"
        score2_surf = score_font.render(score2_text, True, COLORS['snake2'])
        # Add glow effect
        glow_surf = pygame.Surface((score2_surf.get_width() + 10, score2_surf.get_height() + 10), pygame.SRCALPHA)
        glow_surf.blit(score2_surf, (5, 5))
        glow_surf.set_alpha(100)
        self.screen.blit(glow_surf, (self.screen_width - score2_surf.get_width() - 15, self.screen_height - 55))
        self.screen.blit(score2_surf, (self.screen_width - score2_surf.get_width() - 10, self.screen_height - 50))

        # Episode and steps info
        episode_text = f"Episode: {self.episode}"
        steps_text = f"Steps: {self.steps}"
        episode_surf = info_font.render(episode_text, True, COLORS['text'])
        steps_surf = info_font.render(steps_text, True, COLORS['text'])

        # Center episode and steps info
        self.screen.blit(episode_surf, (self.screen_width//2 - episode_surf.get_width()//2, self.screen_height - 55))
        self.screen.blit(steps_surf, (self.screen_width//2 - steps_surf.get_width()//2, self.screen_height - 30))
>>>>>>> 4264bc1 (Added pause menu, improved snake controls, and enhanced visuals)

        # Draw death messages and update leaderboard if snakes have died
        if (hasattr(self.env, 'snake1_death_reason') and self.env.snake1_death_reason) or \
           (hasattr(self.env, 'snake2_death_reason') and self.env.snake2_death_reason):

            # Determine winner and update leaderboard
            if not hasattr(self, '_score_recorded') or not self._score_recorded:
                if snake1_score > snake2_score:
                    winner_color = COLORS['snake1']
                    total_score = snake1_score
                elif snake2_score > snake1_score:
                    winner_color = COLORS['snake2']
                    total_score = snake2_score
                else:  # In case of a tie, give it to P1
                    winner_color = COLORS['snake1']
                    total_score = snake1_score

                self.leaderboard.add_score(
                    self.current_maze_name,
                    winner_color,
                    total_score,
                    snake1_score,
                    snake2_score
                )
                self._score_recorded = True

            # Create a semi-transparent background for death messages
            messages = []
            if self.env.snake1_death_reason:
                message1 = self.death_messages.get(self.env.snake1_death_reason, self.env.snake1_death_reason)
                messages.append(f"Snake 1: {message1}")
            if self.env.snake2_death_reason:
                message2 = self.death_messages.get(self.env.snake2_death_reason, self.env.snake2_death_reason)
                messages.append(f"Snake 2: {message2}")

            if messages:
                total_height = len(messages) * 30 + 140  # Extra height for buttons
                msg_bg = pygame.Surface((self.screen_width - 40, total_height))
                msg_bg.fill((0, 0, 0))
                msg_bg.set_alpha(180)

                msg_y = self.screen_height//2 - total_height//2
                self.screen.blit(msg_bg, (20, msg_y))

                for i, message in enumerate(messages):
                    death_surf = self.font.render(message, True, (255, 50, 50))
                    msg_x = self.screen_width//2 - death_surf.get_width()//2
                    self.screen.blit(death_surf, (msg_x, msg_y + i * 30))

                # Add buttons
                button_width = 200
                button_height = 40
                button_spacing = 20
                button_x = self.screen_width//2 - button_width//2
                button_y = msg_y + len(messages) * 30 + 20

                # Return to Menu button
                pygame.draw.rect(self.screen, COLORS['text'], (button_x, button_y, button_width, button_height))
                pygame.draw.rect(self.screen, COLORS['background'], (button_x + 2, button_y + 2, button_width - 4, button_height - 4))
                button_text = self.font.render("Return to Menu", True, COLORS['text'])
                text_x = button_x + (button_width - button_text.get_width())//2
                text_y = button_y + (button_height - button_text.get_height())//2
                self.screen.blit(button_text, (text_x, text_y))
                self.return_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

                # Select Maze button
                button_y += button_height + button_spacing
                pygame.draw.rect(self.screen, COLORS['text'], (button_x, button_y, button_width, button_height))
                pygame.draw.rect(self.screen, COLORS['background'], (button_x + 2, button_y + 2, button_width - 4, button_height - 4))
                button_text = self.font.render("Select Maze", True, COLORS['text'])
                text_x = button_x + (button_width - button_text.get_width())//2
                text_y = button_y + (button_height - button_text.get_height())//2
                self.screen.blit(button_text, (text_x, text_y))
                self.select_maze_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

        # Draw leaderboard if showing
        if self.show_leaderboard:
            self.leaderboard.render(self.screen, self.current_maze_name, self.font)

    def draw_pause_menu(self):
        """Draw the pause menu with retro punk styling."""
        # Add semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.fill((0, 0, 20))
        overlay.set_alpha(200)
        self.screen.blit(overlay, (0, 0))

        # Draw "PAUSED" text with neon effect
        title_text = "PAUSED"
        title_color = (0, 255, 200)
        title = self.font.render(title_text, True, title_color)
        title_rect = title.get_rect(center=(self.screen_width//2, self.screen_height//4))

        # Add glow effect to title
        glow_surf = pygame.Surface((title.get_width() + 20, title.get_height() + 20), pygame.SRCALPHA)
        temp_surf = self.font.render(title_text, True, title_color)
        temp_surf.set_alpha(50)
        glow_rect = temp_surf.get_rect(center=(glow_surf.get_width()//2, glow_surf.get_height()//2))
        for offset in range(5, 0, -1):
            glow_surf.blit(temp_surf, glow_rect.inflate(offset*2, offset*2))

        self.screen.blit(glow_surf, (title_rect.x - 10, title_rect.y - 10))
        self.screen.blit(title, title_rect)

        # Draw buttons
        for button in self.pause_buttons:
            button.draw(self.screen)

    def render(self):
        """Render the current game state."""
        self.screen.fill(COLORS['background'])
        self.draw_grid()

        # Draw game elements
        if self.env.food and hasattr(self.env.food, 'position'):
            self.draw_cell(self.env.food.position.x, self.env.food.position.y, COLORS['food'], glow=True)
        if self.env.powerup and hasattr(self.env.powerup, 'position'):
            self.draw_cell(self.env.powerup.position.x, self.env.powerup.position.y, (0, 200, 255), glow=True)

        # Draw snakes only if they exist and have bodies
        if self.env.snake2 and hasattr(self.env.snake2, 'body'):
            self.draw_snake(self.env.snake2, COLORS['snake2'])
        if self.env.snake1 and hasattr(self.env.snake1, 'body'):
            self.draw_snake(self.env.snake1, COLORS['snake1'])

        # Draw walls
        if hasattr(self.env.maze, 'barriers'):
            for wall in self.env.maze.barriers:
                self.draw_cell(wall[0], wall[1], COLORS['wall'])

        # Draw UI elements
        self.draw_score()

        if self.paused:
            self.draw_pause_menu()

        pygame.display.flip()

    def render_pause_menu(self):
        """Render the pause menu overlay."""
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # Draw "PAUSED" text
        font = pygame.font.SysFont('arial', 48)
        text = font.render('PAUSED', True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.screen_width//2, self.screen_height//2 - 120))
        self.screen.blit(text, text_rect)

        # Draw buttons
        for button in self.pause_buttons:
            button.draw(self.screen)

    def run(self, num_episodes=1):
        """Run the visualization for a specified number of episodes."""
<<<<<<< HEAD
        menu = self.menu  # Use the existing menu instance

        while True:
            # Update environment with current maze
            maze_path = os.path.join("mazes", menu.maze_files[menu.selected_maze])
            print(f"Loading maze: {maze_path}")
            self.env.update_maze(maze_path)

            # Verify maze was loaded correctly
            if not self.env.maze.is_loaded or self.env.maze.filepath != maze_path:
                print(f"Error: Failed to load maze {maze_path}")
                return

            self.episode = 0
            self.steps = 0
            self.paused = False
            done = False
            return_to_maze_select = False
            pause_screen_needs_update = False

            while not done:
                self.clock.tick(FPS)

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE or event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                            pause_screen_needs_update = True

                    # Handle pause menu buttons only if paused
=======
        running = True
        self._score_recorded = False
        self.game_over = False
        action = 0  # Default action

        # Ensure environment is properly initialized
        obs, _ = self.env.reset()
        print("Initial observation shape:", np.array(obs).shape)

        while running and self.episode < num_episodes:
            self.clock.tick(FPS)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:  # Spacebar toggles pause
                        self.paused = not self.paused
                    elif event.key == pygame.K_ESCAPE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_l:  # 'L' key toggles leaderboard
                        if self.game_over:  # Only allow toggling leaderboard when game is over
                            self.show_leaderboard = not self.show_leaderboard
                    elif event.key == pygame.K_m:  # 'M' key for maze selection
                        self.show_maze_selection()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
>>>>>>> 4264bc1 (Added pause menu, improved snake controls, and enhanced visuals)
                    if self.paused:
                        for button in self.pause_buttons:
                            if button.is_clicked(mouse_pos):
                                if button.text == "Resume":
                                    self.paused = False
                                elif button.text == "Select Maze":
                                    self.show_maze_selection()
                                elif button.text == "Quit":
                                    running = False
                    elif self.game_over:
                        if hasattr(self, 'return_button_rect') and self.return_button_rect.collidepoint(mouse_pos):
                            self.show_maze_selection()
                        elif hasattr(self, 'select_maze_button_rect') and self.select_maze_button_rect.collidepoint(mouse_pos):
                            self.show_maze_selection()

            if not self.paused and not self.show_leaderboard and not self.game_over:
                try:
                    # Get action from model for snake 1
                    if self.model1:
                        # Convert observation to numpy array and ensure correct shape
                        obs_array = np.array(obs, dtype=np.float32)

                        # Debug observation
                        print("Raw observation shape:", obs_array.shape)
                        print("Raw observation range:", obs_array.min(), "-", obs_array.max())

                        # Normalize observation if needed
                        if obs_array.max() > 1.0:
                            obs_array = obs_array / max(SNAKE1_HEAD, SNAKE1_BODY, SNAKE2_HEAD, SNAKE2_BODY)

                        # Ensure correct shape
                        if obs_array.shape != (296,):
                            print(f"Reshaping observation from {obs_array.shape} to (296,)")
                            if len(obs_array.flatten()) < 296:
                                obs_array = np.pad(obs_array.flatten(), (0, 296 - len(obs_array.flatten())), mode='constant')
                            else:
                                obs_array = obs_array.flatten()[:296]

                        # Get valid actions mask
                        valid_actions = self.env._get_valid_actions_mask()
                        print("Valid actions:", valid_actions)

                        try:
                            # Get model prediction
                            action, _ = self.model1.predict(obs_array, deterministic=True)
                            print("Model predicted action:", action)

                            # Validate and clip action
                            action = int(np.clip(action, 0, 3))

                            # If predicted action is invalid and we have valid alternatives, choose a valid one
                            if not valid_actions[action] and any(valid_actions):
                                print(f"Invalid action {action}, choosing from valid actions")
                                valid_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
                                action = random.choice(valid_indices)
                                print("Chose alternative action:", action)

                        except Exception as e:
                            print(f"Error in model prediction: {e}")
                            # Choose random valid action as fallback
                            valid_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
                            action = random.choice(valid_indices) if valid_indices else 0
                            print("Using fallback action:", action)

                    # Step the environment with the validated action
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    print(f"Step result - Action: {action}, Reward: {reward}, Terminated: {terminated}")

                    if info.get("snake1_death_reason"):
                        print("Death reason:", info["snake1_death_reason"])

                    self.steps += 1

                    # Check for game over conditions
                    if terminated or truncated:
                        print("Game over detected")
                        self.death_display_time = time.time()
                        self.game_over = True
                        if self.episode + 1 < num_episodes:
                            print("Starting new episode")
                            self.episode += 1
                            obs, _ = self.env.reset()
                            self.steps = 0
                            self.game_over = False
                            self._score_recorded = False

                except Exception as e:
                    print(f"Critical error during game step: {e}")
                    import traceback
                    traceback.print_exc()
                    self.game_over = True

            # Render the current state
            self.render()

<<<<<<< HEAD
            if return_to_maze_select:
                # Show maze selection menu
                selection = menu.run_maze_only()  # New method to show only maze selection
                if selection is None:  # User quit
                    return
                maze_path = os.path.join("mazes", selection['maze'])
                continue

            # Show game over screen and get next maze
            if self.env.snake2:
                # For two-snake game, determine winner based on death reasons and scores
                snake1_died = bool(self.env.snake1_death_reason)
                snake2_died = bool(self.env.snake2_death_reason)
                snake1_score = self.env.snake1.score
                snake2_score = self.env.snake2.score

                # Determine winner based on death and score
                snake2_wins = (
                    (snake1_died and not snake2_died) or  # Only snake1 died
                    (snake1_died and snake2_died and snake2_score > snake1_score) or  # Both died but snake2 had higher score
                    (not snake1_died and not snake2_died and snake2_score > snake1_score)  # Neither died but snake2 had higher score
                )

                if snake2_wins:
                    # Snake 2 (red) won
                    next_maze = menu.show_game_over(snake2_score, 'red')
                else:
                    # Snake 1 (green) won or it was a draw
                    next_maze = menu.show_game_over(snake1_score, 'green')
            else:
                # Single snake game - always green
                next_maze = menu.show_game_over(self.env.snake1.score, 'green')

            if next_maze is None:
                break
            else:
                # Update maze path for next round
                maze_path = os.path.join("mazes", next_maze['maze'] if isinstance(next_maze, dict) else next_maze)
=======
            # If paused, render the pause menu
            if self.paused:
                self.render_pause_menu()

            # Update the display
            pygame.display.flip()

            # Add a small delay for smooth gameplay
            pygame.time.wait(int(1000/FPS))

        pygame.quit()

    def show_maze_selection(self):
        """Show maze selection menu and reset the game with the selected maze."""
        selection = self.menu.show()
        if selection:
            maze_path = f"mazes/{selection['maze']}"
            self.env.maze.load_maze(maze_path)
            self.current_maze_name = os.path.splitext(os.path.basename(maze_path))[0]
            # Reset game state
            self.episode = 0
            self.steps = 0
            self._score_recorded = False
            self.show_leaderboard = False
            self.paused = False
            self.game_over = False
            # Reset environment
            obs, _ = self.env.reset()
            # Clear death reasons
            self.env.snake1_death_reason = ""
            self.env.snake2_death_reason = ""
            # Reset model if provided
            if 'model' in selection and selection['model']:
                model_path = selection['model']['path']
                try:
                    if isinstance(self.model1, PPO):
                        self.model1 = PPO.load(model_path)
                    elif isinstance(self.model1, DQN):
                        self.model1 = DQN.load(model_path)
                    elif isinstance(self.model1, A2C):
                        self.model1 = A2C.load(model_path)
                except Exception as e:
                    print(f"Error loading new model: {e}")
>>>>>>> 4264bc1 (Added pause menu, improved snake controls, and enhanced visuals)

if __name__ == "__main__":
    try:
        # Initialize environment with default maze
        env = SnakeEnv(
            maze_file="mazes/maze_natural.txt",
            reward_approach="A2",
            opponent_policy='basic_follow'  # This ensures snake2 uses AI
        )

        # Initialize menu with correct dimensions
        screen_width = env.grid_width * CELL_SIZE
        screen_height = env.grid_height * CELL_SIZE + 60
        menu = Menu(screen_width, screen_height)

        # Show model and maze selection menu
        selection = menu.run()
        if selection is None:
            pygame.quit()
            sys.exit(0)

        # Initialize environment with selected maze
        env = SnakeEnv(
            maze_file=f"mazes/{selection['maze']}",
            reward_approach="A2",
            opponent_policy='basic_follow'  # This ensures snake2 uses AI
        )

        # Load the selected model for snake1
        model_path = selection['model']['path']
        model_name = selection['model']['name'].lower()
        print(f"Loading model from: {model_path}")

        # Try to load model metadata first
        try:
            metadata = None
            metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load model metadata: {e}")

        # Determine the algorithm type
        model_type = None
        if metadata and "algorithm" in metadata:
            model_type = metadata["algorithm"]
        else:
            # Fallback to name-based detection
            if 'ppo' in model_name:
                model_type = "PPO"
            elif 'dqn' in model_name:
                model_type = "DQN"
            elif 'a2c' in model_name:
                model_type = "A2C"

        # Load and validate the model
        try:
            if model_type == "PPO":
                model = PPO.load(model_path)
                # Validate observation space matches environment
                if not np.array_equal(model.observation_space.shape, env.observation_space.shape):
                    raise ValueError(f"Model observation space {model.observation_space.shape} does not match environment {env.observation_space.shape}")
            elif model_type == "DQN":
                model = DQN.load(model_path)
                if not np.array_equal(model.observation_space.shape, env.observation_space.shape):
                    raise ValueError(f"Model observation space {model.observation_space.shape} does not match environment {env.observation_space.shape}")
            elif model_type == "A2C":
                model = A2C.load(model_path)
                if not np.array_equal(model.observation_space.shape, env.observation_space.shape):
                    raise ValueError(f"Model observation space {model.observation_space.shape} does not match environment {env.observation_space.shape}")
            else:
                print("Unable to determine model type, defaulting to PPO")
                model = PPO.load(model_path)
                if not np.array_equal(model.observation_space.shape, env.observation_space.shape):
                    raise ValueError(f"Model observation space {model.observation_space.shape} does not match environment {env.observation_space.shape}")
        except Exception as e:
            print(f"Error loading model: {e}")
            pygame.quit()
            sys.exit(1)

        print("Model loaded and validated successfully")

        # Create and run visualizer
        viz = RetroVisualizer(env, model1=model)  # snake1 uses the loaded model, snake2 uses basic_follow AI
        viz.run(num_episodes=5)

    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()
        sys.exit(1)
