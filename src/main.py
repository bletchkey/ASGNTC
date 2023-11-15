import sys
import numpy as np

from games.GameOfLife import GameOfLife
from utils.types import Grid

def main():

    # Create a grid
    random_grid = [[0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1]]

    random_grid = Grid(np.array(random_grid, dtype=int))

    # Initialize the game
    game = GameOfLife(random_grid, simulation_type="toroidal")


    for i in range(33):
        print(f"Step {i+1}:", end="\n")
        game.draw_grid()
        game.update_grid()
        print()

    # Print statistics
    game.print_statistics()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)