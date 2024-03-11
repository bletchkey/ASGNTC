from abc import ABC, abstractmethod

class TrainingBase(ABC):

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _fit(self):
        pass

