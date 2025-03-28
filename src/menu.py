import pygame
import os
import json
from constants import *

class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = (min(color[0] + 30, 255), min(color[1] + 30, 255), min(color[2] + 30, 255))
        self.is_hovered = False

    def draw(self, screen, font):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=12)
        pygame.draw.rect(screen, WHITE, self.rect, 2, border_radius=12)

        text_surface = font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
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

        # Load leaderboard
        self.leaderboard = self.load_leaderboard()

        # States
        self.state = "maze_select"  # "maze_select", "game_over", "leaderboard"
        self.final_score = 0

    def load_leaderboard(self):
        try:
            with open("leaderboard.json", "r") as f:
                return json.load(f)
        except:
            return []

    def save_leaderboard(self):
        with open("leaderboard.json", "w") as f:
            json.dump(self.leaderboard, f)

    def add_score(self, score):
        self.leaderboard.append(score)
        self.leaderboard.sort(reverse=True)
        self.leaderboard = self.leaderboard[:10]  # Keep top 10
        self.save_leaderboard()

    def draw_maze_select(self):
        self.screen.fill((0, 0, 20))  # Dark blue background

        # Draw title
        title = self.title_font.render("Select Maze", True, WHITE)
        title_rect = title.get_rect(center=(self.width//2, 50))
        self.screen.blit(title, title_rect)

        # Draw maze list with descriptions
        for i, maze in enumerate(self.maze_files):
            # Draw maze name
            color = (0, 255, 100) if i == self.selected_maze else WHITE
            name_text = self.font.render(maze, True, color)
            name_rect = name_text.get_rect(center=(self.width//2, 150 + i * 60))
            self.screen.blit(name_text, name_rect)

            # Draw description below name
            desc = self.maze_info.get(maze, "No description available")
            desc_text = pygame.font.SysFont('arial', 24).render(desc, True, (150, 150, 150))
            desc_rect = desc_text.get_rect(center=(self.width//2, 150 + i * 60 + 25))
            self.screen.blit(desc_text, desc_rect)

        # Draw start button
        self.start_button.draw(self.screen, self.font)

    def draw_game_over(self):
        self.screen.fill((0, 0, 20))

        # Draw "Game Over"
        title = self.title_font.render("Game Over", True, WHITE)
        title_rect = title.get_rect(center=(self.width//2, 50))
        self.screen.blit(title, title_rect)

        # Draw score
        score_text = self.font.render(f"Score: {self.final_score}", True, WHITE)
        score_rect = score_text.get_rect(center=(self.width//2, 150))
        self.screen.blit(score_text, score_rect)

        # Draw "View Leaderboard" button
        self.start_button.text = "View Leaderboard"
        self.start_button.draw(self.screen, self.font)

    def draw_leaderboard(self):
        self.screen.fill((0, 0, 20))

        # Draw title
        title = self.title_font.render("Leaderboard", True, WHITE)
        title_rect = title.get_rect(center=(self.width//2, 50))
        self.screen.blit(title, title_rect)

        # Draw scores
        for i, score in enumerate(self.leaderboard):
            text = self.font.render(f"{i+1}. {score}", True, WHITE)
            text_rect = text.get_rect(center=(self.width//2, 150 + i * 40))
            self.screen.blit(text, text_rect)

        # Draw "Play Again" button
        self.start_button.text = "Play Again"
        self.start_button.draw(self.screen, self.font)

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

                if event.type == pygame.KEYDOWN and self.state == "maze_select":
                    if event.key == pygame.K_UP:
                        self.selected_maze = (self.selected_maze - 1) % len(self.maze_files)
                    elif event.key == pygame.K_DOWN:
                        self.selected_maze = (self.selected_maze + 1) % len(self.maze_files)

                if self.start_button.handle_event(event):
                    if self.state == "maze_select":
                        return self.maze_files[self.selected_maze]
                    elif self.state == "game_over":
                        self.state = "leaderboard"
                    else:  # leaderboard
                        self.state = "maze_select"
                        self.start_button.text = "Start Game"

            if self.state == "maze_select":
                self.draw_maze_select()
            elif self.state == "game_over":
                self.draw_game_over()
            else:  # leaderboard
                self.draw_leaderboard()

            pygame.display.flip()
            clock.tick(60)

        return None

    def show_game_over(self, score):
        self.final_score = score
        self.add_score(score)
        self.state = "game_over"
        return self.run()
