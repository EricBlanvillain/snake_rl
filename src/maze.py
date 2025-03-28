# maze.py
import os
import pygame
from constants import GRID_WIDTH, GRID_HEIGHT, WALL, EMPTY

class Maze:
    def __init__(self, filepath=None):
        self.grid = [[EMPTY for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.barriers = set()
        self.filepath = filepath
        self.is_loaded = False  # Track if maze was loaded successfully

        if filepath and os.path.exists(filepath):
            self.load_maze(filepath)
        else:
            print(f"Warning: Maze file '{filepath}' not found or not provided. Using empty grid.")
            # Optional: Add basic boundary walls even if no file
            # self._add_boundary_walls()

    @property
    def width(self):
        return GRID_WIDTH

    @property
    def height(self):
        return GRID_HEIGHT

    def load_maze(self, filepath):
        """Load maze from file and return success status"""
        try:
            if not os.path.exists(filepath):
                print(f"Error: Maze file '{filepath}' does not exist")
                self.is_loaded = False
                return False

            with open(filepath, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print(f"Error: Maze file '{filepath}' is empty")
                    self.is_loaded = False
                    return False

                # Clear existing barriers before loading new ones
                self.grid = [[EMPTY for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
                self.barriers.clear()

                for r, line in enumerate(lines):
                    if r >= GRID_HEIGHT: break
                    line = line.strip()
                    for c, char in enumerate(line):
                        if c >= GRID_WIDTH: break
                        if char == '#':
                            self.grid[r][c] = WALL
                            self.barriers.add((c, r))
                        else:
                            self.grid[r][c] = EMPTY

                # Verify that we loaded at least some walls
                if not self.barriers:
                    print(f"Warning: No walls found in maze file '{filepath}'")

            self.filepath = filepath  # Update filepath after successful load
            self.is_loaded = True
            print(f"Successfully loaded maze from {filepath} with {len(self.barriers)} walls")
            return True
        except Exception as e:
            print(f"Error loading maze from {filepath}: {str(e)}")
            self.is_loaded = False
            return False

    # def _add_boundary_walls(self):
    #     for r in range(GRID_HEIGHT):
    #         for c in range(GRID_WIDTH):
    #             if r == 0 or r == GRID_HEIGHT - 1 or c == 0 or c == GRID_WIDTH - 1:
    #                 if self.grid[r][c] == EMPTY: # Avoid overwriting loaded walls
    #                     self.grid[r][c] = WALL
    #                     self.barriers.add((c, r))

    def is_wall(self, x, y):
        # Check grid boundaries first
        if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
            return True # Treat out of bounds as a wall
        return (x, y) in self.barriers

    def get_random_empty_cell(self, occupied_cells):
        """Finds a random empty cell not occupied by snakes or other items."""
        import random
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if not self.is_wall(x, y) and (x, y) not in occupied_cells:
                return (x, y)

    def draw(self, screen, block_size):
        from constants import GRAY
        for (x, y) in self.barriers:
             # Ensure it's within drawable grid (safety check)
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                pygame.draw.rect(screen, GRAY, (x * block_size, y * block_size, block_size, block_size))

    def rotate_180(self):
        """Rotate the maze 180 degrees."""
        # Create a new rotated grid
        new_grid = [[EMPTY for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        new_barriers = set()

        # Rotate each barrier point
        for (x, y) in self.barriers:
            # For 180-degree rotation:
            # new_x = GRID_WIDTH - 1 - x
            # new_y = GRID_HEIGHT - 1 - y
            new_x = GRID_WIDTH - 1 - x
            new_y = GRID_HEIGHT - 1 - y
            if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT:
                new_barriers.add((new_x, new_y))
                new_grid[new_y][new_x] = WALL

        self.grid = new_grid
        self.barriers = new_barriers

# Example usage:
if __name__ == '__main__':
    # Create a dummy maze file for testing
    maze_file = "mazes/test_maze.txt"
    os.makedirs("mazes", exist_ok=True)
    with open(maze_file, "w") as f:
        f.write("##########\n")
        f.write("#        #\n")
        f.write("#  ####  #\n")
        f.write("#  #  #  #\n")
        f.write("#  #  #  #\n")
        f.write("#     #  #\n")
        f.write("##########\n")

    import pygame
    from constants import BLOCK_SIZE, WIDTH, HEIGHT, WHITE

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Maze Test")

    maze = Maze(maze_file)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        maze.draw(screen, BLOCK_SIZE)
        pygame.display.flip()

    pygame.quit()
