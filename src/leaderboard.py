import json
import os
from datetime import datetime
import pygame
from constants import *

class Leaderboard:
    def __init__(self):
        self.leaderboards = {}
        self.max_entries = 10
        self.load_leaderboards()

    def load_leaderboards(self):
        """Load leaderboards from file, creating it if it doesn't exist."""
        try:
            with open('leaderboards.json', 'r') as f:
                self.leaderboards = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Initialize empty leaderboards for each maze
            maze_files = [f for f in os.listdir("mazes") if f.endswith(".txt")]
            for maze in maze_files:
                maze_name = os.path.splitext(maze)[0]
                self.leaderboards[maze_name] = []
            self.save_leaderboards()

    def save_leaderboards(self):
        """Save leaderboards to file."""
        with open('leaderboards.json', 'w') as f:
            json.dump(self.leaderboards, f, indent=2)

    def add_score(self, maze_name, winner_color, score, snake1_score, snake2_score):
        """Add a new score to the appropriate maze leaderboard."""
        if maze_name not in self.leaderboards:
            self.leaderboards[maze_name] = []

        # Determine winner based on color
        winner = 'P1' if winner_color == COLORS['snake1'] else 'P2'

        entry = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'winner': winner,
            'score': score,
            'snake1_score': snake1_score,
            'snake2_score': snake2_score
        }

        # Insert the new entry in the correct position
        board = self.leaderboards[maze_name]
        board.append(entry)
        board.sort(key=lambda x: x['score'], reverse=True)

        # Keep only top scores
        if len(board) > self.max_entries:
            board = board[:self.max_entries]

        self.leaderboards[maze_name] = board
        self.save_leaderboards()

    def render(self, screen, maze_name, font):
        """Render the leaderboard for a specific maze."""
        if maze_name not in self.leaderboards:
            return

        # Create semi-transparent background
        board_surface = pygame.Surface((500, 350))  # Made larger to fit all info
        board_surface.fill((0, 0, 20))
        board_surface.set_alpha(230)

        # Position the leaderboard in the center of the screen
        x = (screen.get_width() - board_surface.get_width()) // 2
        y = (screen.get_height() - board_surface.get_height()) // 2

        screen.blit(board_surface, (x, y))

        # Draw title with glow effect
        title = font.render(f"Leaderboard - {maze_name}", True, COLORS['text'])
        glow_surf = pygame.Surface((title.get_width() + 10, title.get_height() + 10), pygame.SRCALPHA)
        glow_surf.blit(title, (5, 5))
        glow_surf.set_alpha(100)
        screen.blit(glow_surf, (x + 15, y + 15))
        screen.blit(title, (x + 20, y + 20))

        # Draw headers
        headers = ["Rank", "Winner", "Score", "P1 Score", "P2 Score", "Date"]
        header_y = y + 60
        header_x = x + 20
        spacing = [60, 80, 80, 80, 80, 120]  # Adjusted spacing for each column
        for header, width in zip(headers, spacing):
            text = font.render(header, True, COLORS['score'])
            screen.blit(text, (header_x, header_y))
            header_x += width

        # Draw entries
        entry_y = header_y + 30
        current_x = x + 20
        for i, entry in enumerate(self.leaderboards[maze_name]):
            if i >= self.max_entries:
                break

            current_x = x + 20  # Reset x position for each row

            # Draw rank
            rank_text = font.render(f"#{i+1}", True, COLORS['text'])
            screen.blit(rank_text, (current_x, entry_y))
            current_x += spacing[0]

            # Draw winner (with color)
            winner_color = COLORS['snake1'] if entry['winner'] == 'P1' else COLORS['snake2']
            winner_text = font.render(entry['winner'], True, winner_color)
            screen.blit(winner_text, (current_x, entry_y))
            current_x += spacing[1]

            # Draw total score
            score_text = font.render(str(entry['score']), True, COLORS['text'])
            screen.blit(score_text, (current_x, entry_y))
            current_x += spacing[2]

            # Draw P1 score
            score1_text = font.render(str(entry['snake1_score']), True, COLORS['snake1'])
            screen.blit(score1_text, (current_x, entry_y))
            current_x += spacing[3]

            # Draw P2 score
            score2_text = font.render(str(entry['snake2_score']), True, COLORS['snake2'])
            screen.blit(score2_text, (current_x, entry_y))
            current_x += spacing[4]

            # Draw date
            date_text = font.render(entry['date'], True, COLORS['text'])
            screen.blit(date_text, (current_x, entry_y))

            entry_y += 25
