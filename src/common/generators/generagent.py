import torch

from configs.constants import *


class Generagent():
    """
    Agent that generates initial configurations for the Game of Life.

    The agent needs to create a configuration that once simulated, the associated
    metric is difficult to be predicted by the predictor.


    """

    def __init__(self):
        self.actions = [action for action in range(GRID_SIZE*GRID_SIZE)]

        self.action_probability = torch.ones(GRID_SIZE*GRID_SIZE) / (GRID_SIZE*GRID_SIZE)

    def get_action(self):
        return torch.multinomial(self.action_probability, 1).item()

