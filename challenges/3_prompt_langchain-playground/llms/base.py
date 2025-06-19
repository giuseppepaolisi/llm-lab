from abc import ABC, abstractmethod

class LLMWrapper(ABC):
    @abstractmethod
    def get_model(self):
        pass
