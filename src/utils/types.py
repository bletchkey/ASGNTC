import numpy as np
import hashlib

class Grid(np.ndarray):
    def __new__(cls, data) -> np.ndarray:
        grid = np.asarray(data).view(cls)

        if (grid.shape[0] != grid.shape[1]) or not(grid.ndim == 2) and \
           (grid.shape[0] != 1 and grid.ndim != 3):
            raise ValueError("Grid must be a square matrix.")

        return grid

    def __array_finalize__(self, obj) -> None:
        if obj is None: return

    def __init__(self) -> None:
        super().__init__()
        self.hash = self.__hash__()
        self.size = self.shape[0]

    @property
    def size(self) -> int:
        # return self._size
        return self.shape[0]

    @size.setter
    def size(self, size: int) -> None:
        self._size = size

    @property
    def hash(self) -> int:
        return self._hash

    @hash.setter
    def hash(self, hash: int) -> None:
        self._hash = hash

    def isBinary(self) -> bool:
        return np.all(np.isin(self, [0, 1]))

    def __hash__(self) -> int:
        sha256_hash = hashlib.sha256(self.tobytes()).hexdigest()
        return int(sha256_hash, base=16)


