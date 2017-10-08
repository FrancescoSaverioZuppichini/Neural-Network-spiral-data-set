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

    s = y - t ** 2

    meanCost = np.mean(s)


    ## End
    return meanCost

def dMSE(prediction, target):
    """
    Computes the derivative of the Mean Squared Error function.
    """
    y = prediction
    t = target
    n = prediction.size

    ## Implement


    # error = (y - t) * y*(1 - y)

    ## End
    return (t - y)