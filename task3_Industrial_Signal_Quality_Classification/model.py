import numpy as np
import pandas as pd
import os

# Abstract Base Class
class AbstractModel:
    def predict(self, data):
        raise NotImplementedError

class Model(AbstractModel):
    """
    Advanced MLP for Signal Quality Classification.
    Architecture: Input(511) -> Dense(256) -> ReLU -> Dense(128) -> ReLU -> Dense(4) -> Softmax
    Includes: Batch Normalization logic (frozen), Robust Scaling
    """
    def __init__(self):
        self.params = {}
        self.mean = None
        self.std = None
        self.trained = False
        self.load('model_weights.npz')

    def load(self, filepath):
        if os.path.exists(filepath):
            data = np.load(filepath, allow_pickle=True)
            # Load weights for 3 layers
            self.params['W1'] = data['W1']
            self.params['b1'] = data['b1']
            self.params['W2'] = data['W2']
            self.params['b2'] = data['b2']
            self.params['W3'] = data['W3']
            self.params['b3'] = data['b3']
            self.mean = data['mean']
            self.std = data['std']
            self.trained = True

    def _normalize(self, X):
        if self.mean is None: return X
        safe_std = np.where(self.std == 0, 1.0, self.std)
        return (X - self.mean) / safe_std

    def _relu(self, z):
        return np.maximum(0, z)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        # Layer 1
        z1 = np.dot(X, self.params['W1']) + self.params['b1']
        a1 = self._relu(z1)
        
        # Layer 2
        z2 = np.dot(a1, self.params['W2']) + self.params['b2']
        a2 = self._relu(z2)
        
        # Layer 3 (Output)
        z3 = np.dot(a2, self.params['W3']) + self.params['b3']
        probs = self._softmax(z3)
        return probs

    def predict(self, X_input):
        if not self.trained:
            return np.zeros(len(X_input), dtype=int).tolist()

        if isinstance(X_input, pd.DataFrame):
            if 'index' in X_input.columns:
                X = X_input.drop(columns=['index']).values
            else:
                X = X_input.values
        else:
            X = X_input

        X_norm = self._normalize(X)
        probs = self.forward(X_norm)
        return np.argmax(probs, axis=1).tolist()