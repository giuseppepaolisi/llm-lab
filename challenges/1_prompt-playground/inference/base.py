from abc import ABC, abstractmethod
""" 
Classe base per l'inferenza di modelli LLM"""
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass