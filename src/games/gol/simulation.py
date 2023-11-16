import numpy as np
import hashlib

from utils.types import Grid

_states = {1: "not stable", 0: "stable"}


class Simulation():

    def __init__(self, data=None, simulation_type="standard"):

        if simulation_type not in ["standard", "toroidal"]:
            raise ValueError("The simulation type must be either 'standard' or 'toroidal'.")
        else:
            self.__simulation_type = simulation_type

        if data is None:
            self.__grid = Grid(np.zeros((10, 10), dtype=int))
        else:
            try:
                assert isinstance(data, Grid) and data.dtype == int
                self.__grid = data

                for r, c in np.ndindex(self.__grid.size, self.__grid.size):
                    if not(self.__grid[r, c] == 0 or self.__grid[r, c] == 1):
                        raise ValueError("The grid can only contain 0s and 1s of type int.")

            except AssertionError:
                print("The input must be of type Grid.")

        self.__simulation_step = 0
        self.__simulation_state = 1
        self.__simulation_weights = np.zeros((self.__grid.size, self.__grid.size), dtype=int)
        self.__grid_hashes = [hashlib.sha256(self.__grid.tobytes()).hexdigest()]


    @property
    def grid(self) -> Grid:
        return self.__grid

    @property
    def weights(self) -> np.ndarray:
        return self.__simulation_weights

    @property
    def type(self) -> str:
        return self.__simulation_type

    @property
    def step(self) -> int:
        return self.__simulation_step

    @property
    def state(self) -> str:
        return _states[self.__simulation_state]


    def properties(self) -> dict:
        return {
            "type": self.__simulation_type,
            "step": self.__simulation_step,
            "state": _states[self.__simulation_state]
        }


    def draw_grid(self):
        try:
            assert self.__grid is not None

            for row in self.__grid:
                print(" ".join(map(str, row)))

        except AssertionError:
            print("The grid must be initialized first.")


    def __conways_rules(self, r, c, total_neighbors):
        if self.__grid[r, c] == 1 and (total_neighbors < 2 or total_neighbors > 3):
            return 0
        elif (self.__grid[r, c] == 1 and 2 <= total_neighbors <= 3) or (self.__grid[r, c] == 0 and total_neighbors == 3):
            return 1
        else:
            return self.__grid[r, c]


    def __get_toroidal_neighbors(self, x, y):
        neighbors = []
        for i in [-1, 0, 1]:
            neighbors_row = []
            for j in [-1, 0, 1]:
                wrapped_x = ((x + i) + self.__grid.size) % self.__grid.size
                wrapped_y = ((y + j) + self.__grid.size) % self.__grid.size
                neighbors_row.append(self.__grid[wrapped_x, wrapped_y])
            neighbors.append(neighbors_row)
        return neighbors


    def update_grid(self, steps=1):
        try:
            assert steps > 0 and isinstance(steps, int)

            for _ in range(steps):
                new_grid = self.__grid.copy()
                for r, c in np.ndindex((self.__grid.size, self.__grid.size)):
                    if self.__simulation_type == "toroidal":
                        neighbors = self.__get_toroidal_neighbors(r, c)
                    else:
                        neighbors = self.__grid[r-1:r+2, c-1:c+2]
                    total_neighbors = np.sum(neighbors) - self.__grid[r, c]
                    new_grid[r, c] = self.__conways_rules(r, c, total_neighbors)
                self.__grid = new_grid


            self.__simulation_step += steps

            param = np.power(0.9, self.__simulation_step) * 0.1
            self.__simulation_weights = np.add(self.__simulation_weights, np.multiply(self.__grid, param))

            hash = hashlib.sha256(self.__grid.tobytes()).hexdigest()

            if hash in self.__grid_hashes:
                self.__simulation_state = 0
            else:
                self.__grid_hashes.append(hash)
                self.__simulation_state = 1

            return self.__grid

        except AssertionError:
            print("The number of steps must be greater than 0 and of type int.")
