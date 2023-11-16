import numpy as np

class Grid(np.ndarray):
    def __new__(cls, input_array, grid_property=None):
        obj = np.asarray(input_array).view(cls)

        # Check if the array is a square matrix
        if (obj.shape[0] != obj.shape[1]) or not(obj.ndim == 2):
            raise ValueError("Grid must be a square matrix.")

        obj.grid_property = grid_property
        return obj

    @property
    def size(self) -> int:
        return self.shape[0]

