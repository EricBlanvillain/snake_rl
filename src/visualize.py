# visualize.py
import pygame
import os
import time
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
import sys

from snake_env import SnakeEnv, Direction, Point
from constants import *
from menu import Menu
from button import Button

# Constants for visualization
CELL_SIZE = 30
GRID_LINE_WIDTH = 1
FPS = 15

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
        self.death_display_time = 0
        self.death_messages = {
            "collision_wall": "Hit a wall!",
            "collision_self": "Self collision!",
            "collision_opponent": "Hit opponent snake!",
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
        score_text = f"Score: {self.env.snake1.score if self.env.snake1 else 0}"
        if self.env.snake2:
            score_text += f" vs {self.env.snake2.score}"
        episode_text = f"Episode: {self.episode}"
        steps_text = f"Steps: {self.steps}"

        # Create shaded background for text
        text_bg = pygame.Surface((self.screen_width, 60))
        text_bg.fill(COLORS['background'])
        text_bg.set_alpha(200)
        self.screen.blit(text_bg, (0, self.screen_height - 60))

        # Render text with glow effect
        score_surf = self.font.render(score_text, True, COLORS['score'])
        episode_surf = self.font.render(episode_text, True, COLORS['text'])
        steps_surf = self.font.render(steps_text, True, COLORS['text'])

        self.screen.blit(score_surf, (10, self.screen_height - 55))
        self.screen.blit(episode_surf, (self.screen_width//2 - episode_surf.get_width()//2, self.screen_height - 55))
        self.screen.blit(steps_surf, (self.screen_width - steps_surf.get_width() - 10, self.screen_height - 55))

        # Draw death reason if it exists and hasn't timed out (5 seconds)
        if self.death_reason and time.time() - self.death_display_time < 5:
            # Create a semi-transparent background for death message
            message = self.death_messages.get(self.death_reason, self.death_reason)
            death_surf = self.font.render(f"Game Over: {message}", True, (255, 50, 50))
            msg_bg = pygame.Surface((death_surf.get_width() + 20, death_surf.get_height() + 10))
            msg_bg.fill((0, 0, 0))
            msg_bg.set_alpha(180)

            # Position in center of screen
            msg_x = self.screen_width//2 - death_surf.get_width()//2
            msg_y = self.screen_height//2 - death_surf.get_height()//2

            # Draw background and text with glow effect
            self.screen.blit(msg_bg, (msg_x - 10, msg_y - 5))

            # Add glow effect
            glow_surf = pygame.Surface((death_surf.get_width() + 20, death_surf.get_height() + 20), pygame.SRCALPHA)
            temp_surf = self.font.render(f"Game Over: {message}", True, (255, 50, 50))
            temp_surf.set_alpha(50)
            glow_rect = temp_surf.get_rect(center=(glow_surf.get_width()//2, glow_surf.get_height()//2))
            for offset in range(5, 0, -1):
                glow_surf.blit(temp_surf, glow_rect.inflate(offset*2, offset*2))

            self.screen.blit(glow_surf, (msg_x - 10, msg_y - 10))
            self.screen.blit(death_surf, (msg_x, msg_y))

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
            button.draw(self.screen, self.font)

    def render(self):
        """Render the current game state."""
        self.screen.fill(COLORS['background'])
        self.draw_grid()

        # Draw walls
        for wall in self.env.maze.barriers:
            self.draw_cell(wall[0], wall[1], COLORS['wall'])

        # Draw food with glow
        if self.env.food:
            self.draw_cell(self.env.food.position.x, self.env.food.position.y, COLORS['food'], glow=True)

        # Draw snakes
        if self.env.snake1:
            self.draw_snake(self.env.snake1, COLORS['snake1'])
        if self.env.snake2:
            self.draw_snake(self.env.snake2, COLORS['snake2'])

        self.draw_score()
        pygame.display.flip()

    def run(self, num_episodes=1):
        """Run the visualization for a specified number of episodes."""
        menu = Menu(self.screen_width, self.screen_height)

        while True:
            # Get maze and model selection from menu
            selection = menu.run()
            if selection is None:  # User quit
                return

            # Extract maze path and model info from selection
            maze_path = os.path.join("mazes", selection['maze'])
            model_info = selection['model']

            # Update environment with new maze
            self.env.update_maze(maze_path)

            # Load the selected model
            print(f"Loading model from: {model_info['path']}")
            model_name = model_info['name'].lower()

            # Determine the algorithm type from the model name
            if 'ppo' in model_name:
                self.model1 = PPO.load(model_info['path'])
            elif 'dqn' in model_name:
                self.model1 = DQN.load(model_info['path'])
            elif 'a2c' in model_name:
                self.model1 = A2C.load(model_info['path'])
            else:
                # Default to PPO if unable to determine
                print("Unable to determine model type from name, defaulting to PPO")
                self.model1 = PPO.load(model_info['path'])

            print("Model loaded successfully")

            self.episode = 0
            self.steps = 0
            self.paused = False
            done = False
            return_to_maze_select = False

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

                # Handle pause menu buttons
                if self.paused:
                    for button in self.pause_buttons:
                        if button.handle_event(event):
                            if button.text == "Resume":
                                self.paused = False
                            elif button.text == "Select Maze":
                                return_to_maze_select = True
                                done = True
                                break
                            elif button.text == "Quit":
                                pygame.quit()
                                return

                if return_to_maze_select:
                    break

                if self.paused:
                    self.render()
                    self.draw_pause_menu()
                    pygame.display.flip()
                    continue

                # Get actions from models
                action1 = self.model1.predict(self.env._get_obs(), deterministic=True)[0] if self.model1 else 0

                if self.model2:
                    obs2 = self.env._get_obs()  # Both snakes use the same observation for now
                    action2 = self.model2.predict(obs2, deterministic=True)[0]
                    obs, reward, terminated, truncated, info = self.env.step((action1, action2))
                    done = terminated or truncated
                    if done:
                        self.death_reason = info.get('snake1_death_reason', 'Unknown')
                        self.death_display_time = time.time()
                else:
                    obs, reward, terminated, truncated, info = self.env.step(action1)
                    done = terminated or truncated
                    if done:
                        self.death_reason = info.get('snake1_death_reason', 'Unknown')
                        self.death_display_time = time.time()

                self.steps += 1
                self.render()
                pygame.display.flip()

                # If game is over, show the death message for a moment before continuing
                if done:
                    time.sleep(2)  # Show death message for 2 seconds before continuing

            if return_to_maze_select:
                # Show maze selection menu
                selection = self.menu.run()
                if selection is None:  # User quit
                    pygame.quit()
                    return

                # Extract maze path and model info from selection
                maze_path = os.path.join("mazes", selection['maze'])
                model_info = selection['model']

                # Update environment with new maze
                self.env.update_maze(maze_path)

                # Load the selected model
                print(f"Loading model from: {model_info['path']}")
                model_name = model_info['name'].lower()

                # Determine the algorithm type from the model name
                if 'ppo' in model_name:
                    self.model1 = PPO.load(model_info['path'])
                elif 'dqn' in model_name:
                    self.model1 = DQN.load(model_info['path'])
                elif 'a2c' in model_name:
                    self.model1 = A2C.load(model_info['path'])
                else:
                    # Default to PPO if unable to determine
                    print("Unable to determine model type from name, defaulting to PPO")
                    self.model1 = PPO.load(model_info['path'])

                print("Model loaded successfully")
                continue

            # Show game over screen and get next maze
            next_maze = self.menu.show_game_over(self.env.snake1.score)
            if next_maze is None:
                break
            else:
                # Update environment with new maze
                maze_path = os.path.join("mazes", next_maze['maze'] if isinstance(next_maze, dict) else next_maze)
                self.env.update_maze(maze_path)

if __name__ == "__main__":
    try:
        # Initialize environment with default maze
        env = SnakeEnv(
            maze_file="mazes/maze_natural.txt",
            reward_approach=True,
            opponent_policy='basic_follow'
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
            reward_approach=True,
            opponent_policy='basic_follow'
        )

        # Load the selected model
        model_path = selection['model']['path']
        model_name = selection['model']['name'].lower()
        print(f"Loading model from: {model_path}")

        # Determine the algorithm type from the model name
        if 'ppo' in model_name:
            model = PPO.load(model_path)
        elif 'dqn' in model_name:
            model = DQN.load(model_path)
        elif 'a2c' in model_name:
            model = A2C.load(model_path)
        else:
            # Default to PPO if unable to determine
            print("Unable to determine model type from name, defaulting to PPO")
            model = PPO.load(model_path)

        print("Model loaded successfully")

        # Create and run visualizer
        viz = RetroVisualizer(env, model1=model)
        viz.run(num_episodes=5)

    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()
        sys.exit(1)
