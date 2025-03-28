# maze.py
import os
import pygame
from constants import GRID_WIDTH, GRID_HEIGHT, WALL, EMPTY

class Maze:
    def __init__(self, filepath=None):
        self.grid = [[EMPTY for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.barriers = set()
        if filepath and os.path.exists(filepath):
            self.load_maze(filepath)
        else:
            print(f"Warning: Maze file '{filepath}' not found or not provided. Using empty grid.")
            # Optional: Add basic boundary walls even if no file
            # self._add_boundary_walls()


    def load_maze(self, filepath):
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for r, line in enumerate(lines):
                    # Ensure we don't read more lines than GRID_HEIGHT
                    if r >= GRID_HEIGHT: break
                    line = line.strip()
                    for c, char in enumerate(line):
                         # Ensure we don't read more chars than GRID_WIDTH
                        if c >= GRID_WIDTH: break
                        if char == '#':
                            self.grid[r][c] = WALL
                            self.barriers.add((c, r)) # Store as (x, y)
                        else:
                            self.grid[r][c] = EMPTY
            print(f"Loaded maze from {filepath}")
        except Exception as e:
            print(f"Error loading maze from {filepath}: {e}")
            # self._add_boundary_walls() # Fallback?

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

    def rotate_90(self):
        """Rotate the maze 90 degrees clockwise."""
        # Create a new rotated grid
        new_grid = [[EMPTY for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        new_barriers = set()

        # Rotate each barrier point
        for (x, y) in self.barriers:
            # For 90-degree clockwise rotation:
            # new_x = y
            # new_y = GRID_WIDTH - 1 - x
            new_x = y
            new_y = GRID_WIDTH - 1 - x
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
