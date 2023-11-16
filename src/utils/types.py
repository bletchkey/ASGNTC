import numpy as np
import hashlib

class Grid(np.ndarray):
    def __new__(cls, input_array) -> np.ndarray:
        obj = np.asarray(input_array).view(cls)

        if (obj.shape[0] != obj.shape[1]) or not(obj.ndim == 2):
            raise ValueError("Grid must be a square matrix.")

        return obj

    @property
    def size(self) -> int:
        return self.shape[0]

    def toNumpy(self) -> np.ndarray:
        return np.asarray(self)

    def __hash__(self) -> int:
        sha256_hash = hashlib.sha256(self.tobytes()).hexdigest()
        return int(sha256_hash, 16)


simulation_types = {
    "toroidal" : 0,
    "block" : 1
}