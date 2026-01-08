
import numpy as np
from base_class import BaseMLModel

class Model(BaseMLModel):
    def __init__(self):
        super().__init__()
        self.weights = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Model':
        X_arr = np.array(X)
        y_arr = np.array(y).flatten()
        
        # Add bias term (column of ones)
        ones = np.ones((X_arr.shape[0], 1))
        X_b = np.c_[ones, X_arr]
        
        # Normal Equation: w = (X^T X)^-1 X^T y
        self.weights = np.linalg.pinv(X_b) @ y_arr
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.array(X)
        ones = np.ones((X_arr.shape[0], 1))
        X_b = np.c_[ones, X_arr]
        
        if self.weights is None:
            return np.zeros(len(X))
            
        return X_b @ self.weights
