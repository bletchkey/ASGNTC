from abc import ABC, abstractmethod

class TrainingBase(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _fit(self):
        pass

