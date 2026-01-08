
from abc import ABC, abstractmethod
import numpy as np

class BaseMLModel(ABC):
    def __init__(self): pass
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseMLModel': pass
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: pass
