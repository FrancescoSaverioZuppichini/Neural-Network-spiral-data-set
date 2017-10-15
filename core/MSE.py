import numpy as np

def MSE(prediction, target):
    """
    Computes the Mean Squared Error of a prediction and its target
    """
    y = prediction
    t = target
    n = prediction.size

    if (y.size != n):
        raise Exception("Parameters must have the same len!")

    s = (1 / ( 2 * n )) * np.sum(np.square(y - t))

    ## End
    return s

def dMSE(prediction, target):
    """
    Computes the derivative of the Mean Squared Error function.
    """
    y = prediction
    t = target
    n = prediction.size

    ## End
    return (1 / n) * (y - t)