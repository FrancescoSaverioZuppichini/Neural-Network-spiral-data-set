import numpy as np
import activation as act
import MSE as cost_func
from utils import timing
import MSE as cost
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

class NeuralNetwork:
    """
    Keeps track of the variables of the Multi Layer Perceptron model. Can be
    used for predictoin and to compute the gradients.
    """
    def __init__(self):
        """
        The variables are stored inside a dictonary to make them easy accessible.
        """
        ## Implement
        # hold the layers size
        L_sizes = [7,4]

        # will hold all the intermediate quantity
        self.Z = []
        # will hold all the activation functions
        self.A = []

        # according to this http://cs231n.github.io/neural-networks-2/
        # bias must be small, so let's scale them
        scale_factor = 0.01

        self.var = {
         "W1": np.random.randn(2,20)/np.sqrt(2.0),
         "b1": np.random.random([1,1]) * scale_factor,
         "W2": np.random.rand(20,15)/np.sqrt(20),
         "b2": np.random.random([1,1]) * scale_factor,
         "W3": np.random.rand(15,1)/np.sqrt(15),
         "b3": np.random.random([1,1]) * scale_factor
        }

        ## End

    def forward(self, inputs):
        """
        Implements the forward pass of the MLP model and returns the prediction y. We need to
        store the current input for the backward function.
        """
        x = self.x = inputs

        p = 0.5

        W1 = self.var['W1']
        b1 = self.var['b1']
        W2 = self.var['W2']
        b2 = self.var['b2']
        W3 = self.var['W3']
        b3 = self.var['b3']

        # prediction at each layer
        # find out each z -> zl = W^l * a^{l-1} + b^l
        z1 = inputs.dot(W1) + b1
        # u1 = (np.random.rand(*z1.shape) < p) / p
        # z1 *= u1
        a1 = np.tanh(z1)

        z2 = a1.dot(W2) + b2
        # u2 = (np.random.rand(*z2.shape) < p) / p
        # z2 *= u2
        a2 = np.tanh(z2)

        z3 =  a2.dot(W3) + b3
        # u3 = (np.random.rand(*z3.shape) < p) / p
        # z3 *= u3
        a3 = act.sigmoid(z3)

        self.Z = [z1,z2,z3]
        self.A = [inputs,a1,a2,a3]

        ## End
        return a3


    def backward(self, error):
        """
        Backpropagates through the model and computes the derivatives. The forward function must be
        run before hand for self.x to be defined. Returns the derivatives without applying them using
        a dictonary similar to self.var.
        """
        x = self.x
        W1 = self.var['W1']
        b1 = self.var['b1']
        W2 = self.var['W2']
        b2 = self.var['b2']
        W3 = self.var['W3']
        b3 = self.var['b3']

        Z = self.Z
        A = self.A

        dW3 = error * act.dsigmoid(Z[-1])
        db3 = np.mean(dW3)

        dW2 = dW3.dot(W3.T) * act.dtanh(Z[1])
        db2 = np.mean(dW2)

        dW1 = dW2.dot(W2.T) * act.dtanh(Z[0])
        db1 = np.mean(dW1)

        # compute grads
        dW3 = A[2].T.dot(dW3)
        dW2 = A[1].T.dot(dW2)
        dW1 = A[0].T.dot(dW1)

        ## End
        updates = {"W1": dW1,
                   "b1": db1,
                   "W2": dW2,
                   "b2": db2,
                   "W3": dW3,
                   "b3": db3}

        return updates

    @timing
    def train(self,inputs,targets,learning_rate=0.01, max_iter=200):
        grads = []
        errors = []
        y = 0

        for n in range(max_iter):

            y = self.forward(inputs)

            error = y - targets

            updates = self.backward(error)

            # if (n % 100 == 1):
                # print(cost.MSE(y, targets))
                # print('Error: ', np.mean(np.abs(error)))

            errors.append(np.mean(np.abs(error)))

            for var_str, delta in updates.items():
                update = learning_rate * delta
                self.var[var_str] -= update

            if (cost_func.MSE(y, targets) < 0.02):
                print("DIOCAENE {}".format(cost_func.MSE(y, targets)))
                break
        # print('Error: ',np.mean(np.abs(y)))

        return y, grads, errors