import numpy as np
from activation import sigmoid
from MSE import dMSE
class Perceptron:
    """
    Keeps track of the variables of the Perceptron model. Can be used for predictoin and to compute the gradients.
    """
    def __init__(self):
        """
        The variables are stored inside a dictonary to make them easy accessible.
        """
        self.var = {
         "W": np.array([[.8], [-.5]]),
         "b": 2
        }

    def forward(self, inputs):
        """
        Implements the forward pass of the perceptron model and returns the prediction y. We need to 
        store the current input for the backward function.
        """
        x = self.x = inputs
        W = self.var['W']
        b = self.var['b']

        ## Implement
        y = sigmoid(np.vdot(W, x)+b)


        ## End
        return y

    def backward(self, error):
        """
        Backpropagates through the model and computes the derivatives. The forward function must be 
        run before hand for self.x to be defined. Returns the derivatives without applying them using
        a dictonary similar to self.var.  
        """
        x = self.x
        db = error

        ## Implement
        # dW = np.array([])
        # for i in range(x.size):
        #     # dW[i] = error*x[i]
        #     dW = np.append(dW,error*x[i])

        ## End
        updates = {"W": x*error,
                   "b": db}
        return updates