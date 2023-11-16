
import numpy as np

from utils.types import Grid

class Pattern:

    def load(self, grid, pattern, position=(0, 0)):
        try:
            assert isinstance(grid, Grid)
            assert grid.shape[0] - position[0] >= pattern.shape[0] and grid.shape[1] - position[1] >= pattern.shape[1]

            grid[position[0]:position[0]+pattern.shape[0], position[1]:position[1]+pattern.shape[1]] = pattern

            return grid

        except AssertionError:
            print("The input must be of type Grid and the dimensions must be >= than the pattern dimensions.")
