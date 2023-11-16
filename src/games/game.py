from abc import ABC, abstractmethod

from utils.graphics import Drawable
from utils.types import Grid

class Game(ABC, Drawable):

    @abstractmethod
    def update(self, steps : int) -> Grid:
        pass

    @abstractmethod
    def statistics(self) -> dict:
        pass
