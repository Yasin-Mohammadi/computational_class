import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_dem(size: int = 20) -> np.ndarray:
    """
    Create a synthetic DEM on a size x size grid.

    h[i, j] = 100 + 30*i + 20*j  (meters)
    i, j = 0, 1, ..., size-1

    Elevation increases towards the top-right.
    """
    i = np.arange(size).reshape(-1, 1)  # column vector
    j = np.arange(size).reshape(1, -1)  # row vector
    dem = 100 + 30 * i + 20 * j
    return dem.astype(float)


def create_initial_urban(size: int = 20) -> np.ndarray:
    """
    Create an initial urban configuration on a size x size grid.

    Use a 3x3 urban seed in the bottom-left corner (low elevation region).
    This is big enough that some surrounding cells will have 3 neighbours
    and will start to grow under the given rules.
    """
    urban = np.zeros((size, size), dtype=int)
    # 3x3 block at (0..2, 0..2)
    urban[0:3, 0:3] = 1
    return urban

def count_urban_neighbours(urban: np.ndarray) -> np.ndarray:
    """
    Count number of urban neighbours (Moore neighbourhood, radius 1)
    for each cell using periodic boundary conditions.
    """
    neighbours = np.zeros_like(urban, dtype=int)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbours += np.roll(np.roll(urban, dx, axis=0), dy, axis=1)
    return neighbours


def step_city_growth(urban: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """
    Perform one CA step for city growth with DEM constraints.

    Rules (for non-urban cells):
      - If 5 or more of 8 neighbours are urban AND height < 800  -> becomes urban
      - If 3 or 4 neighbours are urban AND height < 500          -> becomes urban
    Once urban, always urban.
    """
    neighbours = count_urban_neighbours(urban)

    # Conditions for new urbanisation
    cond_high_density = (neighbours >= 5) & (dem < 800)
    cond_medium_density = ((neighbours == 3) | (neighbours == 4)) & (dem < 500)

    # Cells that become urban this step
    new_urban_cells = (urban == 0) & (cond_high_density | cond_medium_density)

    # Urban persists once it appears
    next_urban = urban.copy()
    next_urban[new_urban_cells] = 1

    return next_urban


def show_initial_state(dem: np.ndarray, urban: np.ndarray):
    """
    Show DEM and initial urban pattern in one figure:
    - Left: DEM
    - Right: urban mask
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ax_dem, ax_urban = axes

    im_dem = ax_dem.imshow(dem, origin="lower")
    ax_dem.set_title("DEM (elevation)")
    ax_dem.set_xticks([])
    ax_dem.set_yticks([])
    fig.colorbar(im_dem, ax=ax_dem, fraction=0.046, pad=0.04, label="m")

    im_urban = ax_urban.imshow(urban, origin="lower", cmap="gray_r")
    ax_urban.set_title("Initial urban pattern")
    ax_urban.set_xticks([])
    ax_urban.set_yticks([])

    fig.tight_layout()
    plt.show()


def run_simulation(dem: np.ndarray, initial_urban: np.ndarray,
                   n_steps: int = 50, gif_filename: str = "city_growth.gif"):
    """
    Run the city growth simulation, animate it and save as GIF.

    Requires the 'pillow' writer installed for Matplotlib:
    pip install pillow
    """
    urban = initial_urban.copy()

    fig, ax = plt.subplots()
    im = ax.imshow(urban, origin="lower", interpolation="nearest", cmap="gray_r")
    ax.set_title("City Growth CA with DEM Constraint (step 0)")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        nonlocal urban
        urban = step_city_growth(urban, dem)
        im.set_data(urban)
        ax.set_title(f"City Growth CA with DEM Constraint (step {frame})")
        return [im]

    anim = FuncAnimation(
        fig,
        update,
        frames=n_steps,
        interval=200,  # milliseconds between frames
        blit=True
    )

    # Save as GIF (make sure pillow is installed)
    anim.save(gif_filename, writer="pillow", fps=5)
    print(f"Saved GIF to {gif_filename}")

    plt.show()


if __name__ == "__main__":
    size = 20
    dem = create_dem(size=size)
    initial_urban = create_initial_urban(size=size)

    # 1) Show DEM and initial urban pattern
    show_initial_state(dem, initial_urban)

    # 2) Run simulation and save GIF
    run_simulation(dem, initial_urban, n_steps=50, gif_filename="city_growth.gif")

