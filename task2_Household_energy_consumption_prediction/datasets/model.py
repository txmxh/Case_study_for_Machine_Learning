
import numpy as np
from base_class import BaseMLModel

class Model(BaseMLModel):
    def __init__(self, lambda_reg=10.0):
        super().__init__()
        self.weights = None
        self.lambda_reg = lambda_reg
        
    def fit(self, X, y):
        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=float).flatten()
        ones = np.ones((X_arr.shape[0], 1))
        X_b = np.c_[ones, X_arr]
        I = np.eye(X_b.shape[1]); I[0,0] = 0
        self.weights = np.linalg.inv(X_b.T @ X_b + self.lambda_reg * I) @ X_b.T @ y_arr
        return self

    def predict(self, X):
        X_arr = np.array(X, dtype=float)
        ones = np.ones((X_arr.shape[0], 1))
        X_b = np.c_[ones, X_arr]
        if self.weights is None: return np.zeros(len(X))
        preds = X_b @ self.weights
        preds[preds < 0] = 0
        return preds
