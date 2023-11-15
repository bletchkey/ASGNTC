import numpy as np

from utils.types import Grid

class GameOfLife():

    def __init__(self, data=None, simulation_type="standard"):

        self.grid = None
        self.gridsize = None
        self.simulation_steps = 0

        if simulation_type not in ["standard", "toroidal"]:
            raise ValueError("The simulation type must be either 'standard' or 'toroidal'.")
        else:
            self.simulation_type = simulation_type

        if data is None:
            self.grid = Grid(np.zeros((10, 10), dtype=int))
            self.gridsize = self.grid.shape[0]
        else:
            try:
                assert isinstance(data, Grid) and data.dtype == int
                self.grid = data
                self.gridsize = self.grid.shape[0]

                for r, c in np.ndindex(self.grid.shape):
                    if not(self.grid[r, c] == 0 or self.grid[r, c] == 1):
                        raise ValueError("The grid can only contain 0s and 1s of type int.")

            except AssertionError:
                print("The input must be of type Grid.")


    def draw_grid(self):
        try:
            assert self.grid is not None

            for row in self.grid:
                print(" ".join(map(str, row)))

        except AssertionError:
            print("The grid must be initialized first.")


    def print_statistics(self):
        pass

    def _conways_rules(self, r, c, total_neighbors):
        if self.grid[r, c] == 1 and (total_neighbors < 2 or total_neighbors > 3):
            return 0
        elif (self.grid[r, c] == 1 and 2 <= total_neighbors <= 3) or (self.grid[r, c] == 0 and total_neighbors == 3):
            return 1
        else:
            return self.grid[r, c]


    def _get_toroidal_neighbors(self, x, y):
        neighbors = []
        for i in [-1, 0, 1]:
            neighbors_row = []
            for j in [-1, 0, 1]:
                wrapped_x = ((x + i) + self.gridsize) % self.gridsize
                wrapped_y = ((y + j) + self.gridsize) % self.gridsize
                neighbors_row.append(self.grid[wrapped_x, wrapped_y])
            neighbors.append(neighbors_row)
        return neighbors


    def update_grid(self, steps=1):
        try:
            assert steps > 0 and isinstance(steps, int)

            for _ in range(steps):
                new_grid = self.grid.copy()
                for r, c in np.ndindex((self.gridsize, self.gridsize)):
                    if self.simulation_type == "toroidal":
                        neighbors = self._get_toroidal_neighbors(r, c)
                    else:
                        neighbors = self.grid[r-1:r+2, c-1:c+2]
                    total_neighbors = np.sum(neighbors) - self.grid[r, c]
                    new_grid[r, c] = self._conways_rules(r, c, total_neighbors)
                self.grid = new_grid
            return self.grid


        except AssertionError:
            print("The number of steps must be greater than 0 and of type int.")
