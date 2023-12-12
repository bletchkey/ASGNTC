import sys
import numpy as np

from games import GameOfLife
from games.gameoflife.utils.types import GameOfLifeGrid
from games.gameoflife.utils.patterns import spaceships, oscillators, still_lifes

from utils.graphics import Image, UI
import utils.constants as constants

from advsys.training import Training


def test_game_of_life():

    # Initialize the grid
    initial_grid = GameOfLifeGrid(np.zeros((constants.DEFAULT_GRID_SIZE, constants.DEFAULT_GRID_SIZE), dtype=int), constants.TOPOLOGY["toroidal"])
    initial_grid.load(spaceships["glider"], (2, 2))
    initial_grid.load(spaceships["glider"], (2, 10))
    initial_grid.load(spaceships["glider"], (2, 18))
    initial_grid.load(spaceships["glider"], (10, 2))
    initial_grid.load(oscillators["pulsar"], (10, 10))

    # Initialize random grid
    random_grid = GameOfLifeGrid(np.random.randint(2, size=(constants.DEFAULT_GRID_SIZE, constants.DEFAULT_GRID_SIZE)),
                                 constants.TOPOLOGY["toroidal"])

    random_grid_flat = GameOfLifeGrid(np.random.randint(2, size=(constants.DEFAULT_GRID_SIZE, constants.DEFAULT_GRID_SIZE)),
                                 constants.TOPOLOGY["flat"])

    # Create a new simulation
    # game = GameOfLife(initial_grid)
    game = GameOfLife(random_grid_flat)

    # Run the simulation
    game.update(steps=1000)

    # Print the statistics
    print(game.statistics())

    image_slider = UI(game.all_grids, game.all_weights)
    image_slider.show()


def main():

    train = Training()
    train.run()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)

