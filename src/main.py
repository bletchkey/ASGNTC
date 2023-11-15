import sys
import numpy as np

from games.GameOfLife import GameOfLife
from utils.types import Grid

def main():

    # Create a grid
    initial_grid = [[0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1]]

    initial_grid = Grid(np.array(initial_grid, dtype=int))

    # Initialize the game
    simulation = GameOfLife.Simulation(initial_grid, simulation_type="toroidal")

    for i in range(33):
        print(f"Step {i+1}:", end="\n")
        simulation.draw_grid()
        simulation.update_grid()
        print()

    # Print statistics
    simulation.print_statistics()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)