import numpy as np

from games.game import Game
from .simulation import Simulation

from utils.types import Grid, simulation_types
from utils.graphics import Drawable

class GameOfLifeGame(Game):
    def __init__(self, data : Grid = Grid(np.zeros((10, 10), dtype=int)), simulation_type :int = simulation_types["toroidal"]):
        try:
            assert isinstance(data, Grid)
            self.__simulation = Simulation(data, simulation_type)

        except AssertionError:
            print("The data must be of type Grid.")


    def __iter__(self):
        return self.grid.__iter__()


    def __next__(self):
        return self.grid.__next__()


    @property
    def grid(self) -> Grid:
        return self.__simulation.getGrid()


    @property
    def weights(self) -> np.ndarray:
        return self.__simulation.getWeights()

    @property
    def step(self) -> int:
        return self.__simulation.getSteps()


    @property
    def reached_stability(self) -> int:
        for i in range(len(self.__simulation.stability)):
            if self.__simulation.stability[i]["stable"]:
                return self.__simulation.stability[i]["step"]
        return -1


    def update(self, steps=1) -> Grid:
        try:
            assert steps > 0 and isinstance(steps, int)
            for _ in range(steps):
               self.__simulation.step()

            return self.grid

        except AssertionError:
            print("The number of steps must be a positive integer.")


    def statistics(self) -> dict:
        return {
            "stable" : self.reached_stability,
            "steps"  : self.step,
        }