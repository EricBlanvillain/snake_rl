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

        # Initialize Pygame and font system
        pygame.init()
        pygame.font.init()  # Explicitly initialize font system
        pygame.display.set_caption("Snake RL - Retro Punk Edition")

        # Calculate window dimensions
        self.width = env.grid_width * CELL_SIZE
        self.height = env.grid_height * CELL_SIZE + 60  # Extra space for score

        # Initialize menu
        self.menu = Menu(self.width, self.height)

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

        self.screen = pygame.display.set_mode((self.width, self.height))

    def draw_grid(self):
        """Draw the game grid with a retro effect."""
        for x in range(0, self.width, CELL_SIZE):
            pygame.draw.line(self.screen, COLORS['grid'], (x, 0), (x, self.height - 60), GRID_LINE_WIDTH)
        for y in range(0, self.height - 60, CELL_SIZE):
            pygame.draw.line(self.screen, COLORS['grid'], (0, y), (self.width, y), GRID_LINE_WIDTH)

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
        """Draw score and episode info with retro style."""
        score_text = f"Score: {self.env.snake1.score if self.env.snake1 else 0}"
        if self.env.snake2:
            score_text += f" vs {self.env.snake2.score}"
        episode_text = f"Episode: {self.episode}"
        steps_text = f"Steps: {self.steps}"

        # Create shaded background for text
        text_bg = pygame.Surface((self.width, 60))
        text_bg.fill(COLORS['background'])
        text_bg.set_alpha(200)
        self.screen.blit(text_bg, (0, self.height - 60))

        # Render text with glow effect
        score_surf = self.font.render(score_text, True, COLORS['score'])
        episode_surf = self.font.render(episode_text, True, COLORS['text'])
        steps_surf = self.font.render(steps_text, True, COLORS['text'])

        self.screen.blit(score_surf, (10, self.height - 55))
        self.screen.blit(episode_surf, (self.width//2 - episode_surf.get_width()//2, self.height - 55))
        self.screen.blit(steps_surf, (self.width - steps_surf.get_width() - 10, self.height - 55))

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

    def run(self, num_episodes=5):
        """Run the visualization for specified number of episodes."""
        try:
            for episode in range(num_episodes):
                self.episode = episode + 1
                obs, _ = self.env.reset()
                done = False
                self.steps = 0

                while not done:
                    self.clock.tick(FPS)

                    # Handle events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                pygame.quit()
                                return
                            elif event.key == pygame.K_SPACE:
                                self.paused = not self.paused

                    if self.paused:
                        self.render()
                        continue

                    # Get actions from models
                    action1 = self.model1.predict(obs, deterministic=True)[0] if self.model1 else 0

                    if self.model2:
                        obs2 = self.env._get_obs_for_snake2()
                        action2 = self.model2.predict(obs2, deterministic=True)[0]
                        obs, reward, terminated, truncated, info = self.env.step((action1, action2))
                        done = terminated or truncated
                    else:
                        obs, reward, terminated, truncated, info = self.env.step(action1)
                        done = terminated or truncated

                    self.steps += 1
                    self.render()

                # Show game over screen and get next maze
                next_maze = self.menu.show_game_over(self.env.snake1.score)
                if next_maze is None:
                    break
                else:
                    # Update environment with new maze
                    self.env = SnakeEnv(
                        maze_file=f"mazes/{next_maze}",
                        reward_approach=True,
                        opponent_policy='basic_follow'
                    )

        except Exception as e:
            print(f"Visualization error: {e}")
            pygame.quit()

if __name__ == "__main__":
    try:
        # Initialize menu
        env = SnakeEnv(
            maze_file="mazes/maze_natural.txt",
            reward_approach=True,
            opponent_policy='basic_follow'
        )
        menu = Menu(env.grid_width * CELL_SIZE, env.grid_height * CELL_SIZE + 60)

        # Show maze selection menu
        selected_maze = menu.run()
        if selected_maze is None:
            pygame.quit()
            sys.exit(0)

        # Initialize environment with selected maze
        env = SnakeEnv(
            maze_file=f"mazes/{selected_maze}",
            reward_approach=True,
            opponent_policy='basic_follow'
        )

        # Load the trained model
        model_path = "runs/baseline_dqn_20250328_131808/models/final_model.zip"
        print(f"Loading model from: {model_path}")
        model = DQN.load(model_path)
        print("Model loaded successfully")

        # Create and run visualizer
        viz = RetroVisualizer(env, model1=model)
        viz.run(num_episodes=5)

    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()
        sys.exit(1)
