import numpy as np


def log_loss(y, y_hat):
    m = y.shape[1]
    return - (np.dot(y, np.log(y_hat).T) + np.dot((1 - y), np.log(1 - y_hat).T)) / m


def log_loss_prime(y, y_hat):
    return ((np.divide((1 - y), (1 - y_hat))) - (np.divide(y, y_hat)))

def cross_entropy_loss(t,p):
    t = np.float_(t)
    p = np.float_(p)
    return -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))

def cross_entropy_prime_loss(t,p):
    m = p.shape[0]  # Number of examples
    # Calculate cross-entropy prime
    gradient = (1 / m) * (p - t)
    return gradient