import numpy as np

from .types import GameOfLifeGrid
import utils.constants as constants


spaceships = {
    "glider": GameOfLifeGrid(np.array([[0, 1, 0],
                                       [0, 0, 1],
                                       [1, 1, 1]], dtype=int), topology=constants.TOPOLOGY["flat"]),

    "lightweight_spaceship": GameOfLifeGrid(np.array([[0, 1, 0, 0, 1],
                                                      [1, 0, 0, 0, 0],
                                                      [1, 0, 0, 0, 1],
                                                      [1, 1, 1, 1, 0],
                                                      [0, 0, 0, 0, 0]], dtype=int), topology=constants.TOPOLOGY["flat"]),

    "middleweight_spaceship": GameOfLifeGrid(np.array([[0, 1, 0, 0, 1],
                                                       [1, 0, 0, 0, 0],
                                                       [1, 0, 0, 0, 1],
                                                       [1, 1, 1, 1, 0],
                                                       [0, 0, 0, 0, 1]], dtype=int), topology=constants.TOPOLOGY["flat"]),

    "heavyweight_spaceship": GameOfLifeGrid(np.array([[0, 1, 0, 0, 1, 0],
                                                      [1, 0, 0, 0, 0, 0],
                                                      [1, 0, 0, 0, 1, 0],
                                                      [1, 1, 1, 1, 0, 0],
                                                      [0, 0, 0, 0, 1, 0],
                                                      [0, 1, 1, 1, 0, 0]], dtype=int), topology=constants.TOPOLOGY["flat"])

}


oscillators = {
    "blinker": GameOfLifeGrid(np.array([[0, 0, 0],
                                        [1, 1, 1],
                                        [0, 0, 0]], dtype=int), topology=constants.TOPOLOGY["flat"]),

    "toad": GameOfLifeGrid(np.array([[0, 0, 0, 0],
                                     [0, 1, 1, 1],
                                     [1, 1, 1, 0],
                                     [0, 0, 0, 0]], dtype=int), topology=constants.TOPOLOGY["flat"]),

    "beacon": GameOfLifeGrid(np.array([[1, 1, 0, 0],
                                       [1, 1, 0, 0],
                                       [0, 0, 1, 1],
                                       [0, 0, 1, 1]], dtype=int), topology=constants.TOPOLOGY["flat"]),

    "pulsar": GameOfLifeGrid(np.array([[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
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
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=int), topology=constants.TOPOLOGY["flat"]),

    "pentadecathlon": GameOfLifeGrid(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                                               [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
                                               [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int), topology=constants.TOPOLOGY["flat"])

}


still_lifes = {
    "block": GameOfLifeGrid(np.array([[1, 1],
                                      [1, 1]], dtype=int), topology=constants.TOPOLOGY["flat"]),

    "beehive": GameOfLifeGrid(np.array([[0, 0, 0, 0],
                                        [0, 1, 1, 0],
                                        [1, 0, 0, 1],
                                        [0, 1, 1, 0]], dtype=int), topology=constants.TOPOLOGY["flat"]),

    "loaf": GameOfLifeGrid(np.array([[0, 1, 1, 0],
                                     [1, 0, 0, 1],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0]], dtype=int), topology=constants.TOPOLOGY["flat"]),

    "boat": GameOfLifeGrid(np.array([[1, 1, 0],
                                     [1, 0, 1],
                                     [0, 1, 0]], dtype=int), topology=constants.TOPOLOGY["flat"]),

    "tub": GameOfLifeGrid(np.array([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]], dtype=int), topology=constants.TOPOLOGY["flat"]),
}

