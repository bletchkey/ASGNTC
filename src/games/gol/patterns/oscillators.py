import numpy as np

from .patterns import Pattern

class Blinker:

    def __init__(self):
        self.pattern = np.array([[0, 0, 0],
                                 [1, 1, 1],
                                 [0, 0, 0]], dtype=int)

    def load(self, grid, position=(0, 0)):
        return Pattern.load(self, grid, self.pattern, position)

class Toad:
    def __init__(self):
        self.pattern = np.array([[0, 0, 0, 0],
                                 [0, 1, 1, 1],
                                 [1, 1, 1, 0],
                                 [0, 0, 0, 0]], dtype=int)

    def load(self, grid, position=(0, 0)):
        return Pattern.load(self, grid, self.pattern, position)

class Beacon:
    def __init__(self):
        self.pattern = np.array([[1, 1, 0, 0],
                                 [1, 1, 0, 0],
                                 [0, 0, 1, 1],
                                 [0, 0, 1, 1]], dtype=int)

    def load(self, grid, position=(0, 0)):
        return Pattern.load(self, grid, self.pattern, position)

class Pulsar:
    def __init__(self):
        self.pattern = np.array([[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                 [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                                 [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                                 [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                                 [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                                 [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                                 [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                                 [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=int)

    def load(self, grid, position=(0, 0)):
        return Pattern.load(self, grid, self.pattern, position)


class Pentadecathlon:
    def __init__(self):
        self.pattern = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                                 [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
                                 [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int)

    def load(self, grid, position=(0, 0)):
        return Pattern.load(self, grid, self.pattern, position)