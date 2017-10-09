import numpy as np

def MSE(prediction, target):
    """
    Computes the Mean Squared Error of a prediction and its target
    """
    y = prediction
    t = target
    n = prediction.size

    ## Implement
    meanCost = sum(t - y.reshape(-1,1))**2
    meanCost /= 2*len(prediction)



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

    #error = (1/n)*sum(y-t)
    error = t-y

    ## End
    return error