import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def count_neighbours(grid: np.ndarray) -> np.ndarray:
    """
    Count live neighbours for each cell using periodic boundary conditions.
    grid: 2D array of 0s and 1s
    returns: 2D array of neighbour counts
    """
    # Sum of shifted grids in 8 neighbour directions
    neighbours = np.zeros_like(grid, dtype=int)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbours += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
    return neighbours

def step(grid: np.ndarray) -> np.ndarray:
    """
    Compute one time step of Conway's Game of Life.
    """
    neighbours = count_neighbours(grid)

    # Apply Game of Life rules
    birth = (grid == 0) & (neighbours == 3)
    survive = (grid == 1) & ((neighbours == 2) | (neighbours == 3))

    new_grid = np.zeros_like(grid)
    new_grid[birth | survive] = 1
    return new_grid

def run_simulation(initial_grid: np.ndarray, n_steps: int = 100):
    """
    Run the simulation and animate it using matplotlib.
    """
    grid = initial_grid.copy()

    fig, ax = plt.subplots()
    im = ax.imshow(grid, interpolation="nearest")
    ax.set_title("Conway's Game of Life")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        nonlocal grid
        grid = step(grid)
        im.set_data(grid)
        return [im]

    anim = FuncAnimation(
        fig,
        update,
        frames=n_steps,
        interval=100,  # milliseconds between frames
        blit=True
    )

    plt.show()

def make_blinker(size: int = 20) -> np.ndarray:
    """
    Create a grid with a blinker pattern in the center.
    """
    grid = np.zeros((size, size), dtype=int)
    c = size // 2
    grid[c, c - 1:c + 2] = 1
    return grid

def make_glider(size: int = 30) -> np.ndarray:
    """
    Create a grid with a glider pattern.
    """
    grid = np.zeros((size, size), dtype=int)
    # Glider pattern near top-left corner
    pattern1 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])

    pattern2 = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    pattern3 = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])

    pattern4 = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])

    grid[1:4, 1:4] = pattern1
    grid[5:8, 5:8] = pattern2
    grid[11:14, 11:14] = pattern3
    grid[11:14, 1:4] = pattern4

    return grid

if __name__ == "__main__":
    # Choose an initial configuration:
    # initial = make_blinker(size=30)
    initial = make_glider(size=40)

    run_simulation(initial, n_steps=200)

