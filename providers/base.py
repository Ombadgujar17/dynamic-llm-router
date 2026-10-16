from abc import ABC, abstractmethod

class BaseProvider(ABC):
    @abstractmethod
    def generate(self, model: str, prompt: str) -> dict:
        raise NotImplementedError
