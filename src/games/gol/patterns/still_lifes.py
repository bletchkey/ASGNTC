import numpy as np

from .patterns import Pattern

class Block:
    def __init__(self):
        self.pattern = np.array([[1, 1],
                                 [1, 1]], dtype=int)

    def load(self, grid, position=(0, 0)):
        return Pattern.load(self, grid, self.pattern, position)


class Beehive:
    def __init__(self):
        self.pattern = np.array([[0, 0, 0, 0],
                                 [0, 1, 1, 0],
                                 [1, 0, 0, 1],
                                 [0, 1, 1, 0]], dtype=int)

    def load(self, grid, position=(0, 0)):
        return Pattern.load(self, grid, self.pattern, position)


class Loaf:
    def __init__(self):
        self.pattern = np.array([[0, 1, 1, 0],
                                 [1, 0, 0, 1],
                                 [0, 1, 0, 1],
                                 [0, 0, 1, 0]], dtype=int)

    def load(self, grid, position=(0, 0)):
        return Pattern.load(self, grid, self.pattern, position)


class Boat:
    def __init__(self):
        self.pattern = np.array([[1, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]], dtype=int)

    def load(self, grid, position=(0, 0)):
        return Pattern.load(self, grid, self.pattern, position)


class Tub:
    def __init__(self):
        self.pattern = np.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]], dtype=int)

    def load(self, grid, position=(0, 0)):
        return Pattern.load(self, grid, self.pattern, position)