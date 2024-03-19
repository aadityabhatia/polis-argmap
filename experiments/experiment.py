from abc import ABC, abstractmethod

class Experiment(ABC):
    requiresLanguageModel = False

    @staticmethod
    @abstractmethod
    def run(self, dataset, languageModel=None):
        pass
