import numpy as np

from games.game import Game
from .simulation import Simulation
from .utils.types import GameOfLifeGrid
import utils.constants as constants

class GameOfLife(Game):
    def __init__(self, data: GameOfLifeGrid = GameOfLifeGrid(np.zeros((constants.DEFAULT_GRID_SIZE, constants.DEFAULT_GRID_SIZE), dtype=int), constants.TOPOLOGY["toroidal"])) -> None:
        try:
            if data.__class__ == GameOfLifeGrid:
                self.__simulation = Simulation(data)

        except Exception as e:
            raise e


    def __iter__(self):
        return self.grid.__iter__()

    def __next__(self):
        return self.grid.__next__()


    @property
    def grid(self) -> GameOfLifeGrid:
        return self.__simulation.getGrid()

    @property
    def weights(self) -> np.ndarray:
        return self.__simulation.getWeights()

    @property
    def step(self) -> int:
        return self.__simulation.getSteps()

    @property
    def all_grids(self) -> list:
        return self.__simulation.getAllGrids()

    @property
    def all_weights(self) -> list:
        return self.__simulation.getAllWeights()

    @property
    def metric(self) -> np.ndarray:
        return self.__simulation.getMetric()

    @property
    def reached_stability(self) -> int:
        for i in range(len(self.__simulation.stability)):
            if self.__simulation.stability[i]["stable"]:
                return self.__simulation.stability[i]["step"]
        return -1


    def update(self, steps=1) -> GameOfLifeGrid:
        try:
            if steps <= 0:
                raise ValueError("The number of steps must be greater than 0.")

            for _ in range(steps):
               self.__simulation.step()

            return self.grid

        except Exception as e:
            raise e


    def statistics(self) -> dict:
        return {
            "stable" : self.reached_stability,
            "steps"  : self.step,
        }