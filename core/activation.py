import math
import numpy as np

def dsigmoid(x):

    sigma = sigmoid(x)
    return sigma * (1.0 - sigma)

def tanh(x):
    """
    Implements the hyperbolic tangent activation function.
    """
    ## Implement
    e_x = math.e ** 2* x

    # End

    return (e_x + 1)/(e_x - 1)



def dtanh(x):
    """
    Implements the derivative of the hyperbolic tangent activation function.
    """
    ## Implement



    ## End
    return 1 - np.tanh(x)**2


def sigmoid(x):

    return 1/(1 + math.e ** -x)