import math
import numpy as np

def dsigmoid(x):

    sigma = sigmoid(x)
    return sigma * (1.0 - sigma)

def tanh(x):
    """
    Implements the hyperbolic tangent activation function.
    """
    return np.tanh(x)

def dtanh(x):
    """
    Implements the derivative of the hyperbolic tangent activation function.
    """
    return 1 - np.tanh(x)**2

def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1. * (x > 0)

def sigmoid(x):
    return 1/(1 + np.exp(-x))