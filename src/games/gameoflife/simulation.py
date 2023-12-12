import numpy as np

from .utils.types import GameOfLifeGrid
import utils.constants as constants

class Simulation:
    def __init__(self, grid: GameOfLifeGrid):

        self.__grids = [{
            "grid"   : grid,
            "hash"   : grid.hash,
            "weights": np.zeros((grid.size, grid.size), dtype=float)
        }]

        self.__gridsize = grid.size
        self.__step = 0
        self.__stability = [{"stable": False, "step": None}]

        self.__type = grid.topology


    @property
    def __current_grid(self) -> GameOfLifeGrid:
        return self.__grids[self.__step]["grid"]

    @property
    def __current_grid_hash(self) -> int:
        return self.__grids[self.__step]["hash"]

    @property
    def __current_weights(self) -> np.ndarray:
        return self.__grids[self.__step]["weights"]

    @property
    def stability(self) -> list:
        return self.__stability

    @property
    def gridsize(self) -> int:
        return self.__gridsize


    def getGrid(self) -> GameOfLifeGrid:
        return self.__current_grid

    def getWeights(self) -> np.ndarray:
        return self.__current_weights

    def getSteps(self) -> int:
        return self.__step

    def getAllGrids(self) -> list:
        grids = []
        for grid in self.__grids:
            grids.append(grid["grid"])
        return grids

    def getAllWeights(self) -> list:
        weights = []
        for grid in self.__grids:
            weights.append(grid["weights"])
        return weights


    def step(self) -> GameOfLifeGrid:
        new_grid = self.__current_grid.copy()

        for r, c in np.ndindex(self.__gridsize, self.__gridsize):
            if self.__type == constants.TOPOLOGY["toroidal"]:
                neighbors = self.__get_toroidal_neighbors(r, c)
            else:
                neighbors = self.__current_grid[r-1:r+2, c-1:c+2]

            total_neighbors = np.sum(neighbors) - self.__current_grid[r, c]
            new_grid[r, c] = self.__conways_rules(r, c, total_neighbors)

        self.__update_params(new_grid)

        return self.__current_grid


    def __conways_rules(self, r :int, c :int, total_neighbors :int) -> int:
        if self.__current_grid[r, c] == 1 and (total_neighbors < 2 or total_neighbors > 3):
            return 0
        elif (self.__current_grid[r, c] == 1 and 2 <= total_neighbors <= 3) or (self.__current_grid[r, c] == 0 and total_neighbors == 3):
            return 1
        else:
            return self.__current_grid[r, c]


    def __get_toroidal_neighbors(self, r :int, c :int):
        neighbors = []
        for i in [-1, 0, 1]:
            neighbors_row = []
            for j in [-1, 0, 1]:
                wrapped_r = ((r + i) + self.__gridsize) % self.__gridsize
                wrapped_c = ((c + j) + self.__gridsize) % self.__gridsize
                neighbors_row.append(self.__current_grid[wrapped_r, wrapped_c])
            neighbors.append(neighbors_row)
        return neighbors

    def __calculate_weights(self) -> np.ndarray:
        parameter = np.power(0.999, self.__step) * 0.1
        new_weights = np.add(self.__current_weights, np.multiply(self.__current_grid, parameter))
        return new_weights

    def __update_params(self, new_grid :GameOfLifeGrid) -> None:
        self.__grids.append({
            "grid"   : new_grid,
            "hash"   : hash(new_grid),
            "weights": self.__calculate_weights()
        })

        self.__step += 1

        if any(grid["hash"] == self.__current_grid_hash for grid in self.__grids[:-1]):
            self.__stability.append({"stable": True, "step": self.__step})



