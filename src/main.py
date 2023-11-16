import sys
import numpy as np

from games.gameoflife.game import GameOfLifeGame
from games.gameoflife.patterns.oscillators import Blinker, Toad, Beacon, Pulsar, Pentadecathlon
from games.gameoflife.patterns.spaceships import Glider, LightWeightSpaceship, MiddleWeightSpaceship, HeavyWeightSpaceship
from games.gameoflife.patterns.still_lifes import Beehive, Block, Boat, Loaf, Tub

from utils.types import Grid, simulation_types

def main():

    # Initialize the grid
    initial_grid = Grid(np.zeros((32, 32), dtype=int))

    initial_grid = Glider().load(initial_grid, (0, 0))
    initial_grid = LightWeightSpaceship().load(initial_grid, (20, 10))
    initial_grid = Beacon().load(initial_grid, (10, 10))
    initial_grid = Pulsar().load(initial_grid, (0, 20))
    initial_grid = Pentadecathlon().load(initial_grid, (20, 20))
    initial_grid = Blinker().load(initial_grid, (10, 0))
    initial_grid = Toad().load(initial_grid, (0, 10))
    initial_grid = Block().load(initial_grid, (20, 0))
    initial_grid = Glider().load(initial_grid, (0, 20))

    # Create a new simulation
    game = GameOfLifeGame(initial_grid, simulation_types["toroidal"])

    game.draw()
    game.update(steps=200)
    print()
    game.draw()
    print()
    print(game.statistics())

if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)

