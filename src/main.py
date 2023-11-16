import sys
import numpy as np
import matplotlib.pyplot as plt

from games.gol.simulation import Simulation
from games.gol.patterns.oscillators import Blinker, Toad, Beacon, Pulsar, Pentadecathlon
from games.gol.patterns.spaceships import Glider, LightWeightSpaceship, MiddleWeightSpaceship, HeavyWeightSpaceship
from games.gol.patterns.still_lifes import Beehive, Block, Boat, Loaf, Tub

from utils.display import Image
from utils.types import Grid


def main():

    # Initialize the grid
    # initial_grid = Grid(np.zeros((32, 32), dtype=int))
    # initial_grid = Glider().load(initial_grid, (0, 0))
    # initial_grid = LightWeightSpaceship().load(initial_grid, (20, 10))
    # initial_grid = Beacon().load(initial_grid, (10, 10))
    # initial_grid = Pulsar().load(initial_grid, (0, 20))
    # initial_grid = Pentadecathlon().load(initial_grid, (20, 20))
    # initial_grid = Blinker().load(initial_grid, (10, 0))
    # initial_grid = Toad().load(initial_grid, (0, 10))
    # initial_grid = Block().load(initial_grid, (20, 0))
    # initial_grid = Glider().load(initial_grid, (0, 20))

    initial_grid = Grid(np.zeros((32, 32), dtype=int))
    initial_grid = Glider().load(initial_grid, (0, 0))

    # Create a new simulation
    simulation = Simulation(initial_grid, simulation_type="toroidal")


    for _ in range(100):
        simulation.update_grid()

    print()
    simulation.draw_grid()
    print()

    weights_image = Image(simulation.weights)
    weights_image.show()

    # image = Image(simulation.grid)
    # image.show()

    # Show the properties of the simulation
    for i in simulation.properties():
        print(f"{i}: {simulation.properties()[i]}")

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)