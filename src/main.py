import sys
import numpy as np

from games import GameOfLife
from games.gameoflife.utils.types import GameOfLifeGrid
from games.gameoflife.utils.patterns import spaceships, oscillators, still_lifes

from utils.graphics import Image, GUI
import utils.constants as constants

def main():

    # Initialize the grid
    initial_grid = GameOfLifeGrid(np.zeros((constants.DEFAULT_GRID_SIZE, constants.DEFAULT_GRID_SIZE), dtype=int), constants.TOPOLOGY["toroidal"])
    initial_grid.load(spaceships["glider"], (2, 2))
    initial_grid.load(spaceships["lightweight_spaceship"], (20, 20))
    initial_grid.load(oscillators["blinker"], (5, 5))
    initial_grid.load(oscillators["beacon"], (10, 10))
    initial_grid.load(still_lifes["beehive"], (20, 20))


    # Create a new simulation
    game = GameOfLife(initial_grid)

    # Run the simulation
    game.update(steps=500)

    # Print the statistics
    print(game.statistics())

    image_slider = GUI(game.all_grids, game.all_weights)
    image_slider.show()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)

