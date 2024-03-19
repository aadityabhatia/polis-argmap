from abc import ABC, abstractmethod

class Task(ABC):
    requiresLanguageModel = False

    @staticmethod
    @abstractmethod
    def run(self, dataset, languageModel=None):
        pass
