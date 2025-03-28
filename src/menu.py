import pygame
import os
import json
from constants import *
import numpy as np
import time
from datetime import datetime

class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = (min(color[0] + 30, 255), min(color[1] + 30, 255), min(color[2] + 30, 255))
        self.is_hovered = False
        self.glow_amount = 0
        self.glow_direction = 1

    def draw(self, screen, font):
        # Update glow animation
        self.glow_amount += 0.1 * self.glow_direction
        if self.glow_amount >= 1.0:
            self.glow_direction = -1
        elif self.glow_amount <= 0.0:
            self.glow_direction = 1

        color = self.hover_color if self.is_hovered else self.color

        # Create button surface with glow
        button_surf = pygame.Surface((self.rect.width + 20, self.rect.height + 20))
        button_surf.fill((0, 0, 0))  # Fill with black for transparency
        button_surf.set_colorkey((0, 0, 0))  # Make black transparent

        # Draw multiple rectangles for glow effect
        glow_alpha = int(50 * self.glow_amount) if self.is_hovered else 30
        for i in range(4, 0, -1):
            glow_rect = pygame.Rect(10-i, 10-i, self.rect.width+i*2, self.rect.height+i*2)
            glow_surface = pygame.Surface((glow_rect.width, glow_rect.height))
            glow_surface.fill((0, 0, 0))
            glow_surface.set_colorkey((0, 0, 0))
            pygame.draw.rect(glow_surface, color, glow_surface.get_rect(), border_radius=12)
            glow_surface.set_alpha(glow_alpha)
            button_surf.blit(glow_surface, (glow_rect.x - (10-i), glow_rect.y - (10-i)))

        # Main button rectangle
        main_rect = pygame.Rect(10, 10, self.rect.width, self.rect.height)
        pygame.draw.rect(button_surf, color, main_rect, border_radius=12)

        # Add scanlines effect
        scanline_surface = pygame.Surface((self.rect.width, 2))
        scanline_surface.fill((0, 0, 0))
        scanline_surface.set_alpha(50)
        for y in range(0, self.rect.height, 2):
            button_surf.blit(scanline_surface, (10, 10+y))

        # Border with neon effect
        border_surface = pygame.Surface((main_rect.width, main_rect.height))
        border_surface.fill((0, 0, 0))
        border_surface.set_colorkey((0, 0, 0))
        pygame.draw.rect(border_surface, color, border_surface.get_rect(), 2, border_radius=12)
        button_surf.blit(border_surface, (10, 10))

        # Draw the button surface
        screen.blit(button_surf, (self.rect.x - 10, self.rect.y - 10))

        # Render text with glow
        text_color = (255, 255, 255) if self.is_hovered else (220, 220, 220)
        text_surface = font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)

        # Add text glow
        if self.is_hovered:
            glow_text = text_surface.copy()
            glow_text.set_alpha(50)
            for i in range(3):
                screen.blit(glow_text, (text_rect.x - i, text_rect.y))
                screen.blit(glow_text, (text_rect.x + i, text_rect.y))
                screen.blit(glow_text, (text_rect.x, text_rect.y - i))
                screen.blit(glow_text, (text_rect.x, text_rect.y + i))

        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False

class Menu:
    def __init__(self, screen_width, screen_height):
        self.width = screen_width
        self.height = screen_height

        # Initialize Pygame and display
        pygame.init()
        pygame.font.init()  # Explicitly initialize font system
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Snake RL - Menu")

        # Load fonts after initialization
        try:
            self.title_font = pygame.font.Font(None, 74)
            self.font = pygame.font.Font(None, 36)
        except Exception as e:
            print(f"Error loading fonts: {e}")
            # Fallback to system font if custom font fails
            self.title_font = pygame.font.SysFont('arial', 74)
            self.font = pygame.font.SysFont('arial', 36)

        # Create buttons
        button_width = 200
        button_height = 50
        button_x = (screen_width - button_width) // 2

        self.start_button = Button(button_x, screen_height - 100,
                                 button_width, button_height,
                                 "Start Game", (0, 150, 0))

        # Maze descriptions
        self.maze_info = {
            'maze_natural.txt': 'Natural flowing corridors and open spaces',
            'maze_symmetric.txt': 'Symmetrical design for fair gameplay',
            'maze_arena.txt': 'Open arena with strategic obstacles',
            'maze_training.txt': 'Progressive difficulty for training',
            'maze_rooms.txt': 'Connected rooms with multiple paths'
        }

        # Load maze list
        self.maze_files = [f for f in os.listdir("mazes") if f.endswith(".txt")]
        self.selected_maze = 0

        # Load models list
        self.models = self.scan_models()
        self.selected_model = 0

        # Load leaderboard
        self.leaderboard = self.load_leaderboard()

        # States
        self.state = "model_select"  # "model_select", "maze_select", "game_over", "leaderboard"
        self.final_score = 0

    def scan_models(self):
        """Scan the runs directory for available models."""
        models = []
        runs_dir = "runs"
        if os.path.exists(runs_dir):
            for run_dir in os.listdir(runs_dir):
                model_path = os.path.join(runs_dir, run_dir, "models", "final_model.zip")
                if os.path.exists(model_path):
                    models.append({
                        'name': run_dir,
                        'path': model_path
                    })
        return models

    def load_leaderboard(self):
        """Load leaderboard and convert old format scores if needed"""
        try:
            with open("leaderboard.json", "r") as f:
                data = json.load(f)
                # Initialize per-maze leaderboards if not present
                if isinstance(data, list):
                    # Convert old format (list) to new format (dict with maze types)
                    converted_data = {}
                    for maze_file in self.maze_files:
                        converted_data[maze_file] = []

                    # Add old scores to 'maze_natural.txt' (default maze)
                    for entry in data:
                        if isinstance(entry, int):
                            converted_data['maze_natural.txt'].append({
                                'score': entry,
                                'timestamp': 'Legacy Score'
                            })
                        else:
                            converted_data['maze_natural.txt'].append(entry)
                    return converted_data
                return data
        except Exception as e:
            print(f"Error loading leaderboard: {e}")
            # Initialize empty leaderboards for each maze
            return {maze_file: [] for maze_file in self.maze_files}

    def save_leaderboard(self):
        with open("leaderboard.json", "w") as f:
            json.dump(self.leaderboard, f)

    def add_score(self, score, maze_type, snake_color='green'):
        """Add a new score to the leaderboard with timestamp and snake color"""
        try:
            # Ensure score is an integer
            score_value = int(score)
            score_entry = {
                'score': score_value,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'snake_color': snake_color
            }

            # Initialize maze type if not present
            if maze_type not in self.leaderboard:
                self.leaderboard[maze_type] = []

            self.leaderboard[maze_type].append(score_entry)
            # Sort by score in descending order
            self.leaderboard[maze_type].sort(key=lambda x: x['score'], reverse=True)
            self.leaderboard[maze_type] = self.leaderboard[maze_type][:10]  # Keep top 10
            self.save_leaderboard()
            print(f"Added score {score_value} for maze {maze_type} with snake color {snake_color}")
        except Exception as e:
            print(f"Error adding score: {e}")

    def draw_model_select(self):
        """Draw the model selection screen with retro punk styling."""
        self.screen.fill((0, 0, 20))  # Dark blue background

        # Draw cyberpunk grid effect
        for x in range(0, self.width, 30):
            alpha = abs(int(128 + 64 * np.sin(time.time() * 2 + x * 0.1)))  # Pulsing effect
            line_surf = pygame.Surface((2, self.height), pygame.SRCALPHA)
            # Create a temporary surface for the line
            temp_line = pygame.Surface((2, self.height))
            temp_line.fill((40, 100, 255))
            temp_line.set_alpha(alpha)
            line_surf.blit(temp_line, (0, 0))
            self.screen.blit(line_surf, (x, 0))

        # Draw glowing title with neon effect
        title_text = "SELECT MODEL"
        title_color = (0, 255, 200)
        # Main text
        title = self.title_font.render(title_text, True, title_color)
        title_rect = title.get_rect(center=(self.width//2, 50))

        # Glow effect
        glow_surf = pygame.Surface((title.get_width() + 20, title.get_height() + 20), pygame.SRCALPHA)
        temp_surf = self.title_font.render(title_text, True, title_color)
        temp_surf.set_alpha(50)
        glow_rect = temp_surf.get_rect(center=(glow_surf.get_width()//2, glow_surf.get_height()//2))
        for offset in range(5, 0, -1):
            glow_surf.blit(temp_surf, glow_rect.inflate(offset*2, offset*2))

        self.screen.blit(glow_surf, (title_rect.x - 10, title_rect.y - 10))
        self.screen.blit(title, title_rect)

        # Draw model list with neon effect
        for i, model in enumerate(self.models):
            # Calculate pulsing effect for selected model
            pulse = int(abs(np.sin(time.time() * 4)) * 50)
            base_color = (0, min(200 + pulse, 255), 100) if i == self.selected_model else (200, 200, 200)

            # Draw model name with neon glow
            name_text = self.font.render(model['name'], True, base_color)
            name_rect = name_text.get_rect(center=(self.width//2, 150 + i * 60))

            if i == self.selected_model:
                # Create glow effect for selected item
                glow_surf = pygame.Surface((name_text.get_width() + 20, name_text.get_height() + 20), pygame.SRCALPHA)
                temp_text = self.font.render(model['name'], True, base_color)
                temp_text.set_alpha(30)
                glow_rect = temp_text.get_rect(center=(glow_surf.get_width()//2, glow_surf.get_height()//2))

                for offset in range(3, 0, -1):
                    glow_surf.blit(temp_text, glow_rect.inflate(offset*2, offset*2))
                self.screen.blit(glow_surf, (name_rect.x - 10, name_rect.y - 10))

            self.screen.blit(name_text, name_rect)

            # Draw selection indicator
            if i == self.selected_model:
                indicator_width = 10
                indicator_height = 20
                left_indicator = pygame.Surface((indicator_width, indicator_height))
                right_indicator = pygame.Surface((indicator_width, indicator_height))

                # Set the colorkey for transparency
                left_indicator.set_colorkey((0, 0, 0))
                right_indicator.set_colorkey((0, 0, 0))

                # Fill with black first (for transparency)
                left_indicator.fill((0, 0, 0))
                right_indicator.fill((0, 0, 0))

                # Create triangle shapes
                pygame.draw.polygon(left_indicator, base_color,
                    [(indicator_width, 0), (0, indicator_height//2), (indicator_width, indicator_height)])
                pygame.draw.polygon(right_indicator, base_color,
                    [(0, 0), (indicator_width, indicator_height//2), (0, indicator_height)])

                # Position indicators
                self.screen.blit(left_indicator, (name_rect.left - 30, name_rect.centery - indicator_height//2))
                self.screen.blit(right_indicator, (name_rect.right + 20, name_rect.centery - indicator_height//2))

        # Draw start button with neon effect
        self.start_button.text = "SELECT MODEL"
        self.start_button.color = (0, 150, 255)  # Neon blue
        self.start_button.draw(self.screen, self.font)

        # Draw navigation hint
        hint_text = "↑↓ to navigate • ENTER to select"
        hint_color = (100, 100, 150)
        hint_surf = pygame.font.SysFont('arial', 20).render(hint_text, True, hint_color)
        hint_rect = hint_surf.get_rect(center=(self.width//2, self.height - 40))
        self.screen.blit(hint_surf, hint_rect)

    def draw_maze_select(self):
        self.screen.fill((0, 0, 20))  # Dark blue background

        # Draw title with same style as model select
        title_text = "SELECT MAZE"
        title_color = (0, 255, 200)
        title = self.title_font.render(title_text, True, title_color)
        title_rect = title.get_rect(center=(self.width//2, 50))

        # Add glow effect to title
        glow_surf = pygame.Surface((title.get_width() + 20, title.get_height() + 20), pygame.SRCALPHA)
        temp_surf = self.title_font.render(title_text, True, title_color)
        temp_surf.set_alpha(50)
        glow_rect = temp_surf.get_rect(center=(glow_surf.get_width()//2, glow_surf.get_height()//2))
        for offset in range(5, 0, -1):
            glow_surf.blit(temp_surf, glow_rect.inflate(offset*2, offset*2))

        self.screen.blit(glow_surf, (title_rect.x - 10, title_rect.y - 10))
        self.screen.blit(title, title_rect)

        # Draw maze list with descriptions
        for i, maze in enumerate(self.maze_files):
            # Calculate pulsing effect for selected maze
            pulse = int(abs(np.sin(time.time() * 4)) * 50)
            base_color = (0, min(200 + pulse, 255), 100) if i == self.selected_maze else (200, 200, 200)

            # Draw maze name with glow effect
            name_text = self.font.render(maze, True, base_color)
            name_rect = name_text.get_rect(center=(self.width//2, 150 + i * 60))

            if i == self.selected_maze:
                # Add glow effect for selected maze
                glow_surf = pygame.Surface((name_text.get_width() + 20, name_text.get_height() + 20), pygame.SRCALPHA)
                temp_text = self.font.render(maze, True, base_color)
                temp_text.set_alpha(30)
                glow_rect = temp_text.get_rect(center=(glow_surf.get_width()//2, glow_surf.get_height()//2))
                for offset in range(3, 0, -1):
                    glow_surf.blit(temp_text, glow_rect.inflate(offset*2, offset*2))
                self.screen.blit(glow_surf, (name_rect.x - 10, name_rect.y - 10))

            self.screen.blit(name_text, name_rect)

            # Draw description below name
            desc = self.maze_info.get(maze, "No description available")
            desc_text = pygame.font.SysFont('arial', 24).render(desc, True, (150, 150, 150))
            desc_rect = desc_text.get_rect(center=(self.width//2, 150 + i * 60 + 25))
            self.screen.blit(desc_text, desc_rect)

            # Draw selection indicator
            if i == self.selected_maze:
                indicator_width = 10
                indicator_height = 20
                left_indicator = pygame.Surface((indicator_width, indicator_height), pygame.SRCALPHA)
                right_indicator = pygame.Surface((indicator_width, indicator_height), pygame.SRCALPHA)

                # Create triangle shapes with RGB colors (no alpha)
                pygame.draw.polygon(left_indicator, base_color[:3],
                    [(indicator_width, 0), (0, indicator_height//2), (indicator_width, indicator_height)])
                pygame.draw.polygon(right_indicator, base_color[:3],
                    [(0, 0), (indicator_width, indicator_height//2), (0, indicator_height)])

                # Position indicators
                self.screen.blit(left_indicator, (name_rect.left - 30, name_rect.centery - indicator_height//2))
                self.screen.blit(right_indicator, (name_rect.right + 20, name_rect.centery - indicator_height//2))

        # Draw start button with neon green
        self.start_button.text = "Start Game"
        self.start_button.color = (0, 255, 100)  # Neon green
        self.start_button.draw(self.screen, self.font)

    def draw_game_over(self):
        self.screen.fill((0, 0, 20))

        # Draw "Game Over" with neon effect
        title_text = "GAME OVER"
        title_color = (0, 255, 200)  # Same neon color as other screens
        title = self.title_font.render(title_text, True, title_color)
        title_rect = title.get_rect(center=(self.width//2, 50))

        # Add glow effect to title
        glow_surf = pygame.Surface((title.get_width() + 20, title.get_height() + 20), pygame.SRCALPHA)
        temp_surf = self.title_font.render(title_text, True, title_color)
        temp_surf.set_alpha(50)
        glow_rect = temp_surf.get_rect(center=(glow_surf.get_width()//2, glow_surf.get_height()//2))
        for offset in range(5, 0, -1):
            glow_surf.blit(temp_surf, glow_rect.inflate(offset*2, offset*2))

        self.screen.blit(glow_surf, (title_rect.x - 10, title_rect.y - 10))
        self.screen.blit(title, title_rect)

        # Draw score with neon effect
        score_color = (255, 50, 150)  # Neon pink for score
        score_text = self.font.render(f"Score: {self.final_score}", True, score_color)
        score_rect = score_text.get_rect(center=(self.width//2, 150))

        # Add glow to score
        score_glow = pygame.Surface((score_text.get_width() + 20, score_text.get_height() + 20), pygame.SRCALPHA)
        temp_score = self.font.render(f"Score: {self.final_score}", True, score_color)
        temp_score.set_alpha(30)
        score_glow_rect = temp_score.get_rect(center=(score_glow.get_width()//2, score_glow.get_height()//2))
        for offset in range(3, 0, -1):
            score_glow.blit(temp_score, score_glow_rect.inflate(offset*2, offset*2))

        self.screen.blit(score_glow, (score_rect.x - 10, score_rect.y - 10))
        self.screen.blit(score_text, score_rect)

        # Draw "View Leaderboard" button with neon blue
        self.start_button.text = "View Leaderboard"
        self.start_button.color = (0, 150, 255)  # Neon blue
        self.start_button.draw(self.screen, self.font)

    def draw_leaderboard(self):
        self.screen.fill((0, 0, 20))

        # Draw title with neon effect - use shorter maze name
        maze_name = self.maze_files[self.selected_maze].replace('maze_', '').replace('.txt', '')
        title_text = f"LEADERBOARD - {maze_name}"
        title_color = (0, 255, 200)  # Same neon color as other screens
        title = pygame.font.SysFont('arial', 42).render(title_text, True, title_color)
        title_rect = title.get_rect(center=(self.width//2, 50))

        # Add glow effect to title
        glow_surf = pygame.Surface((title.get_width() + 20, title.get_height() + 20), pygame.SRCALPHA)
        temp_surf = pygame.font.SysFont('arial', 42).render(title_text, True, title_color)
        temp_surf.set_alpha(50)
        glow_rect = temp_surf.get_rect(center=(glow_surf.get_width()//2, glow_surf.get_height()//2))
        for offset in range(5, 0, -1):
            glow_surf.blit(temp_surf, glow_rect.inflate(offset*2, offset*2))

        self.screen.blit(glow_surf, (title_rect.x - 10, title_rect.y - 10))
        self.screen.blit(title, title_rect)

        # Draw navigation hint
        hint_text = "↑↓ to change maze"
        hint_color = (100, 100, 150)
        hint_surf = pygame.font.SysFont('arial', 20).render(hint_text, True, hint_color)
        hint_rect = hint_surf.get_rect(center=(self.width//2, 90))
        self.screen.blit(hint_surf, hint_rect)

        # Draw column headers
        header_y = 120
        header_color = (150, 150, 200)
        rank_header = pygame.font.SysFont('arial', 24).render("RANK", True, header_color)
        score_header = pygame.font.SysFont('arial', 24).render("SCORE", True, header_color)
        snake_header = pygame.font.SysFont('arial', 24).render("SNAKE", True, header_color)
        date_header = pygame.font.SysFont('arial', 24).render("DATE & TIME", True, header_color)

        # Calculate column positions
        rank_x = self.width//5 - 80
        score_x = 2*self.width//5 - 50
        snake_x = 3*self.width//5 - 20
        date_x = 4*self.width//5 + 20

        # Draw headers
        self.screen.blit(rank_header, (rank_x, header_y))
        self.screen.blit(score_header, (score_x, header_y))
        self.screen.blit(snake_header, (snake_x, header_y))
        self.screen.blit(date_header, (date_x - date_header.get_width()//2, header_y))

        # Get current maze's leaderboard
        current_maze = self.maze_files[self.selected_maze]
        maze_scores = self.leaderboard.get(current_maze, [])

        # Draw scores with neon effect
        start_y = header_y + 40
        spacing = 40
        base_score_color = (200, 200, 255)

        for i, entry in enumerate(maze_scores):
            # Add pulsing effect for top 3 scores
            if i < 3:
                pulse = int(abs(np.sin(time.time() * 4)) * 50)
                if i == 0:
                    score_color = (min(200 + pulse, 255), 200, 255)  # Gold
                elif i == 1:
                    score_color = (200, min(200 + pulse, 255), 255)  # Silver
                else:
                    score_color = (200, 200, min(255 + pulse, 255))  # Bronze
            else:
                score_color = base_score_color

            # Render rank
            rank_text = self.font.render(f"#{i+1}", True, score_color)
            rank_rect = rank_text.get_rect(left=rank_x, centery=start_y + i * spacing)

            # Render score
            score_text = self.font.render(f"{entry['score']}", True, score_color)
            score_rect = score_text.get_rect(left=score_x, centery=start_y + i * spacing)

            # Render snake indicator (circle with snake color)
            snake_color = entry.get('snake_color', 'green')  # Default to green for legacy scores
            snake_color_rgb = (0, 255, 100) if snake_color == 'green' else (255, 50, 50)  # Green or Red
            snake_circle = pygame.Surface((24, 24), pygame.SRCALPHA)
            pygame.draw.circle(snake_circle, snake_color_rgb, (12, 12), 10)
            snake_rect = snake_circle.get_rect(centerx=snake_x + 12, centery=start_y + i * spacing)

            # Render date/time
            date_text = pygame.font.SysFont('arial', 24).render(entry['timestamp'], True, (150, 150, 150))
            date_rect = date_text.get_rect(centerx=date_x, centery=start_y + i * spacing)

            # Add glow effect for top 3 scores
            if i < 3:
                for text, rect in [(rank_text, rank_rect), (score_text, score_rect)]:
                    glow_surf = pygame.Surface((text.get_width() + 20, text.get_height() + 20), pygame.SRCALPHA)
                    temp_text = text.copy()
                    temp_text.set_alpha(30)
                    glow_rect = temp_text.get_rect(center=(glow_surf.get_width()//2, glow_surf.get_height()//2))
                    for offset in range(3, 0, -1):
                        glow_surf.blit(temp_text, glow_rect.inflate(offset*2, offset*2))
                    self.screen.blit(glow_surf, (rect.x - 10, rect.y - 10))

            # Draw the text and snake indicator
            self.screen.blit(rank_text, rank_rect)
            self.screen.blit(score_text, score_rect)
            self.screen.blit(snake_circle, snake_rect)
            self.screen.blit(date_text, date_rect)

        # Draw "Play Again" button with neon green
        self.start_button.text = "Play Again"
        self.start_button.color = (0, 255, 100)  # Neon green
        self.start_button.draw(self.screen, self.font)

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

                if event.type == pygame.KEYDOWN:
                    if self.state == "model_select":
                        if event.key == pygame.K_UP:
                            self.selected_model = (self.selected_model - 1) % len(self.models)
                        elif event.key == pygame.K_DOWN:
                            self.selected_model = (self.selected_model + 1) % len(self.models)
                    elif self.state == "maze_select" or self.state == "leaderboard":
                        if event.key == pygame.K_UP:
                            self.selected_maze = (self.selected_maze - 1) % len(self.maze_files)
                        elif event.key == pygame.K_DOWN:
                            self.selected_maze = (self.selected_maze + 1) % len(self.maze_files)

                if self.start_button.handle_event(event):
                    if self.state == "model_select":
                        self.state = "maze_select"
                        self.start_button.text = "Start Game"
                        continue
                    elif self.state == "maze_select":
                        return {
                            'maze': self.maze_files[self.selected_maze],
                            'model': self.models[self.selected_model]
                        }
                    elif self.state == "game_over":
                        self.state = "leaderboard"
                    else:  # leaderboard
                        # Return to maze selection with the same model
                        self.state = "maze_select"
                        self.start_button.text = "Start Game"

            if self.state == "model_select":
                self.draw_model_select()
            elif self.state == "maze_select":
                self.draw_maze_select()
            elif self.state == "game_over":
                self.draw_game_over()
            else:  # leaderboard
                self.draw_leaderboard()

            pygame.display.flip()
            clock.tick(60)

        return None

    def show_game_over(self, score, snake_color='green'):
        """Show game over screen and add score to leaderboard."""
        self.final_score = score
        self.add_score(score, self.maze_files[self.selected_maze], snake_color)
        self.state = "game_over"
        return self.run()

    def run_maze_only(self):
        """Run the menu showing only maze selection."""
        running = True
        clock = pygame.time.Clock()
        self.state = "maze_select"  # Force maze selection state

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected_maze = (self.selected_maze - 1) % len(self.maze_files)
                    elif event.key == pygame.K_DOWN:
                        self.selected_maze = (self.selected_maze + 1) % len(self.maze_files)

                if self.start_button.handle_event(event):
                    return {
                        'maze': self.maze_files[self.selected_maze]
                    }

            # Draw the maze selection screen
            self.draw_maze_select()
            pygame.display.flip()
            clock.tick(60)

        return None
