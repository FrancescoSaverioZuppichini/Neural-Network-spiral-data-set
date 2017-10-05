import math

def dsigmoid(x):

    sigma = sigmoid(x)
    return sigma * (1.0 - sigma)

    ## End
    return x

def tanh(x):
    """
    Implements the hyperbolic tangent activation function.
    """
    ## Implement



    ## End
    return x

def dtanh(x):
    """
    Implements the derivative of the hyperbolic tangent activation function.
    """
    ## Implement



    ## End
    return x


def sigmoid(x):

    return 1/(1 + math.e ** -x)