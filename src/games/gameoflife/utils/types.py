from utils.types import Grid
import utils.constants as constants

class GameOfLifeGrid(Grid):
    def __new__(cls, data: Grid, topology: str) -> Grid:
        grid = super().__new__(cls, data)
        return grid

    def __init__(self, data: Grid, topology: str) -> None:
        super().__init__()

        if not self.isBinary():
            raise ValueError("The Game of Life grid must be binary.")

        if topology not in constants.TOPOLOGY:
            raise ValueError("The topology must be: {}.", constants.TOPOLOGY)

        self.__topology = topology

    @property
    def topology(self) -> str:
        return self.__topology

    """
    position: the tuple (x, y) indicates the position of the grid to be loaded starting from the top left corner.
    """
    def load(self, grid, position=(0, 0)):
        try:
            assert isinstance(grid, self.__class__)

            if (self.shape[0] - (position[0]+grid.shape[0]) >= grid.shape[0] and self.shape[1] - (position[1]+grid.shape[1]) >= grid.shape[1]):
                self[position[0]:position[0]+grid.shape[0], position[1]:position[1]+grid.shape[1]] = grid
            else:
                raise ValueError("The grid can't be loaded in the specified position.")

            return self

        except AssertionError:
            print("The grid to be loaded must be a GameOfLifeGrid")

