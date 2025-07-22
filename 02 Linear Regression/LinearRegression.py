import numpy as np


class LinearRegression:

    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None 
        
    def _standardize(self, X):
        return (X - self.mean) / self.std
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = self._standardize(X)
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        X = self._standardize(X)
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
