# game_elements.py
import pygame
from collections import namedtuple
import random
from constants import (GRID_WIDTH, GRID_HEIGHT, BLOCK_SIZE,
                       GREEN1, BLUE1, GREEN2, BLUE2, RED, YELLOW, BLACK, WHITE)

Point = namedtuple('Point', 'x, y')

class Direction:
    UP = Point(0, -1)
    DOWN = Point(0, 1)
    LEFT = Point(-1, 0)
    RIGHT = Point(1, 0)
    # Map integer actions (0, 1, 2, 3) to directions
    INDEX_TO_DIR = [UP, DOWN, LEFT, RIGHT]
    # Map directions to integer actions
    DIR_TO_INDEX = {UP: 0, DOWN: 1, LEFT: 2, RIGHT: 3}
    # Opposite directions for checking invalid moves
    OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

    @staticmethod
    def get_components(direction):
        """Get the x and y components of a direction."""
        return direction.x, direction.y

class Snake:
    def __init__(self, snake_id, start_pos, start_dir, color1, color2):
        self.id = snake_id
        self.head = Point(*start_pos)
        self.body = [self.head,
                     Point(self.head.x - start_dir.x, self.head.y - start_dir.y),
                     Point(self.head.x - 2 * start_dir.x, self.head.y - 2 * start_dir.y)]
        self.direction = start_dir
        self.color1 = color1
        self.color2 = color2
        self.score = 0
        self.grow_pending = 0
        self.is_dead = False
        self.powerup_active = None # Could store type/duration
        self.steps_since_last_food = 0  # New counter for food timeout
        self.death_reason = ""  # Track reason for death

    def move(self, action):
        # Get direction from action integer (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        new_direction = Direction.INDEX_TO_DIR[action]

        # Prevent moving directly backward
        if len(self.body) > 1 and new_direction == Direction.OPPOSITE.get(self.direction):
             # Keep current direction if trying to reverse
            pass # print(f"Snake {self.id}: Invalid move (reverse), continuing {self.direction}")
        else:
            self.direction = new_direction

        # Update head position
        self.head = Point(self.head.x + self.direction.x, self.head.y + self.direction.y)
        self.body.insert(0, self.head)

        # Handle growing or shrinking
        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            self.body.pop() # Remove tail if not growing

        # Increment steps since last food
        self.steps_since_last_food += 1

    def grow(self, amount=1):
        self.grow_pending += amount
        self.steps_since_last_food = 0  # Reset counter when food is eaten

    def check_collision_self(self):
        # Check if head collides with body (excluding the very next segment if len > 1)
        return self.head in self.body[1:]

    def get_positions(self):
        return set(self.body)

    def draw(self, screen, block_size):
        for i, point in enumerate(self.body):
            color = self.color1 if i == 0 else self.color2
            # Ensure point is within grid bounds before drawing
            if 0 <= point.x < GRID_WIDTH and 0 <= point.y < GRID_HEIGHT:
                 pygame.draw.rect(screen, color, (point.x * block_size, point.y * block_size, block_size, block_size))
                 # Optional: outline head
                 if i == 0:
                      pygame.draw.rect(screen, BLACK, (point.x * block_size, point.y * block_size, block_size, block_size), 1)


class Food:
    def __init__(self, position, points=1):
        self.position = Point(*position)
        self.points = points
        self.color = YELLOW

    def draw(self, screen, block_size):
         pygame.draw.rect(screen, self.color, (self.position.x * block_size, self.position.y * block_size, block_size, block_size))
         pygame.draw.ellipse(screen, RED, (self.position.x * block_size + block_size//4, self.position.y * block_size + block_size//4, block_size//2, block_size//2)) # Inner circle

class PowerUp:
    def __init__(self, position, points=5, type='extra_points'):
        self.position = Point(*position)
        self.points = points # Can be points, duration, etc.
        self.type = type
        self.color = BLUE2

    def draw(self, screen, block_size):
        pygame.draw.rect(screen, self.color, (self.position.x * block_size, self.position.y * block_size, block_size, block_size))
        # Simple 'P' shape in white
        pygame.draw.rect(screen, WHITE, (self.position.x * block_size + block_size//4, self.position.y * block_size + block_size//5, block_size//6, block_size*3//5))
        pygame.draw.rect(screen, WHITE, (self.position.x * block_size + block_size//4, self.position.y * block_size + block_size//5, block_size*2//5, block_size//5))
        pygame.draw.rect(screen, WHITE, (self.position.x * block_size + block_size*3//6, self.position.y * block_size + block_size*2//5, block_size//6, block_size//5))
