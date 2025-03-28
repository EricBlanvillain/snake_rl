# visualize.py
import pygame
import os
import time
import numpy as np
from stable_baselines3 import PPO, DQN, A2C

from snake_env import SnakeEnv
from constants import *

# --- Retro Punk Style Settings ---
BACKGROUND_COLOR = (0, 0, 0)  # Black
GRID_COLOR = (40, 40, 40)  # Dark gray
GRID_HIGHLIGHT = (60, 0, 60)  # Dark purple for neon effect

# Neon Colors
NEON_GREEN = (0, 255, 128)
NEON_PINK = (255, 20, 147)
NEON_BLUE = (0, 191, 255)
NEON_PURPLE = (147, 0, 211)
NEON_YELLOW = (255, 255, 0)

# UI Colors
TEXT_COLOR = (255, 255, 255)  # White
HEADER_COLOR = NEON_PINK
SCORE_COLOR = NEON_GREEN

class RetroVisualizer:
    def __init__(self, model1_path, model2_path=None, fps=10):
        pygame.init()
        pygame.font.init()

        # Load retro fonts
        try:
            self.font = pygame.font.Font("src/assets/Streamster.ttf", 32)
            self.small_font = pygame.font.Font("src/assets/Streamster.ttf", 24)
        except:
            print("Retro font not found, using system font")
            self.font = pygame.font.SysFont('arial', 32)
            self.small_font = pygame.font.SysFont('arial', 24)

        # Calculate dimensions
        self.block_size = BLOCK_SIZE
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.padding = 40  # Padding for UI elements

        # Window dimensions including UI space
        self.window_width = self.grid_width * self.block_size + self.padding * 2
        self.window_height = self.grid_height * self.block_size + self.padding * 4

        # Setup display
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Snake RL - Retro Punk Edition")

        # Load models
        self.model1 = self._load_model(model1_path)
        self.model2 = self._load_model(model2_path) if model2_path else None

        # Setup environment
        self.env = SnakeEnv(render_mode="rgb_array")
        self.clock = pygame.time.Clock()
        self.fps = fps

    def _load_model(self, path):
        """Load a model based on its filename."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        if "ppo" in path.lower():
            return PPO.load(path)
        elif "dqn" in path.lower():
            return DQN.load(path)
        elif "a2c" in path.lower():
            return A2C.load(path)
        else:
            raise ValueError(f"Unknown model type for {path}")

    def _draw_neon_rect(self, surface, color, rect, width=1):
        """Draw a rectangle with a neon glow effect."""
        # Draw multiple rectangles with decreasing alpha for glow effect
        for i in range(3):
            alpha = 255 - i * 50
            s = pygame.Surface((rect[2] + i*2, rect[3] + i*2), pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, alpha), (i, i, rect[2], rect[3]), width)
            surface.blit(s, (rect[0] - i, rect[1] - i))

    def _draw_grid(self):
        """Draw the game grid with neon effect."""
        for x in range(self.grid_width + 1):
            pygame.draw.line(self.screen, GRID_COLOR,
                           (x * self.block_size + self.padding, self.padding),
                           (x * self.block_size + self.padding, self.grid_height * self.block_size + self.padding))
        for y in range(self.grid_height + 1):
            pygame.draw.line(self.screen, GRID_COLOR,
                           (self.padding, y * self.block_size + self.padding),
                           (self.grid_width * self.block_size + self.padding, y * self.block_size + self.padding))

    def _draw_game_header(self, episode, steps, score1, score2=None):
        """Draw the game information header."""
        # Episode info
        episode_text = f"EPISODE {episode}"
        episode_surface = self.font.render(episode_text, True, HEADER_COLOR)
        self.screen.blit(episode_surface, (self.padding, 5))

        # Steps
        steps_text = f"STEPS: {steps}"
        steps_surface = self.small_font.render(steps_text, True, NEON_BLUE)
        self.screen.blit(steps_surface, (self.window_width - 150, 5))

        # Scores
        score_text = f"SNAKE 1: {score1}"
        score_surface = self.small_font.render(score_text, True, NEON_GREEN)
        self.screen.blit(score_surface, (self.padding, self.window_height - 30))

        if score2 is not None:
            score2_text = f"SNAKE 2: {score2}"
            score2_surface = self.small_font.render(score2_text, True, NEON_PINK)
            self.screen.blit(score2_surface, (self.window_width - 200, self.window_height - 30))

    def run_visualization(self, num_episodes=5):
        """Run the visualization for specified number of episodes."""
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            steps = 0

            while not (done or truncated):
                # Clear screen
                self.screen.fill(BACKGROUND_COLOR)

                # Draw grid
                self._draw_grid()

                # Get actions from models
                action1, _ = self.model1.predict(obs, deterministic=True)
                if self.model2:
                    # If we have two models, get second action
                    obs2 = self.env._get_obs_for_snake2()  # You'll need to implement this
                    action2, _ = self.model2.predict(obs2, deterministic=True)
                    obs, reward, done, truncated, info = self.env.step_two_agents(action1, action2)
                else:
                    # Single agent mode
                    obs, reward, done, truncated, info = self.env.step(action1)

                # Draw game elements with neon effect
                if self.env.food:
                    food_rect = (
                        self.env.food.position.x * self.block_size + self.padding,
                        self.env.food.position.y * self.block_size + self.padding,
                        self.block_size,
                        self.block_size
                    )
                    self._draw_neon_rect(self.screen, NEON_YELLOW, food_rect)

                # Draw snake 1 with neon effect
                for i, segment in enumerate(self.env.snake1.body):
                    snake_rect = (
                        segment.x * self.block_size + self.padding,
                        segment.y * self.block_size + self.padding,
                        self.block_size,
                        self.block_size
                    )
                    color = NEON_GREEN if i == 0 else (0, 200, 0)
                    self._draw_neon_rect(self.screen, color, snake_rect)

                # Draw snake 2 if present
                if self.env.snake2:
                    for i, segment in enumerate(self.env.snake2.body):
                        snake_rect = (
                            segment.x * self.block_size + self.padding,
                            segment.y * self.block_size + self.padding,
                            self.block_size,
                            self.block_size
                        )
                        color = NEON_PINK if i == 0 else (200, 20, 147)
                        self._draw_neon_rect(self.screen, color, snake_rect)

                # Draw UI elements
                self._draw_game_header(
                    episode + 1,
                    steps,
                    self.env.snake1.score,
                    self.env.snake2.score if self.env.snake2 else None
                )

                # Update display
                pygame.display.flip()

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
                            # Pause/Unpause
                            paused = True
                            while paused:
                                for event in pygame.event.get():
                                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                                        paused = False
                                    elif event.type == pygame.QUIT:
                                        pygame.quit()
                                        return

                steps += 1
                self.clock.tick(self.fps)

            # Short delay between episodes
            time.sleep(1)

        pygame.quit()

if __name__ == "__main__":
    # Example usage
    model1_path = "runs/baseline_dqn_20250328_130003/models/final_model.zip"  # Your model path
    visualizer = RetroVisualizer(model1_path, fps=10)
    visualizer.run_visualization(num_episodes=5)
