import numpy as np


def MSE_grad(y, y_pred):
    """Derivative of MSE loss with respect to prediction."""
    return (2 / len(y)) * (y_pred - y)


def MAE_grad(y, y_pred):
    """Derivative of MAE loss with subgradient at zero set to zero."""
    return np.sign(y_pred - y) / len(y)


def logloss_sigmoid_grad(y, y_pred):
    """Gradient used by the assignment logistic regression model."""
    return MSE_grad(y, y_pred) / 2
