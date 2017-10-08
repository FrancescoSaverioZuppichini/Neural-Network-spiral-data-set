import numpy as np
import activation as act
import MSE as cost_func
from utils import timing
import math
from concurrent.futures import ThreadPoolExecutor

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
        # store all the delta for each layer
        self.deltas = []
        #W1_in = ...
        #W1_out = ...
        #W2_in = ...
        #W2_out = ...
        #W3_in = ...
        #W3_out = ...

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

    def forward(self, inputs, t):
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
        a_1 = np.tanh(z1)

        z2 = a_1.dot(W2) + b2
        # u2 = (np.random.rand(*z2.shape) < p) / p
        # z2 *= u2
        a_2 = np.tanh(z2)

        z3 =  a_2.dot(W3) + b3
        # u3 = (np.random.rand(*z3.shape) < p) / p
        # z3 *= u3
        a_3 = act.sigmoid(z3)

        self.Z = [z1,z2,z3]
        # self.A = [np.array([inputs]),a_1,a_2,a_3]

        self.A = [inputs,a_1,a_2,a_3]

        ## End
        return a_3

    def update(self, key, dx, learning_rage):
        self.var[key] -= dx * learning_rage

    def backward(self, error, learning_rate):
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

        dW3 = error * act.dsigmoid(Z[-1])
        db3 = np.mean(dW3)


        dW2 = dW3.dot(W3.T) * act.dtanh(Z[1])
        db2 = np.mean(dW2)

        dW1 = dW2.dot(W2.T) * act.dtanh(Z[0])
        db1 = np.mean(dW1)

        # compute grads
        dW3 = self.A[2].T.dot(dW3)
        dW2 = self.A[1].T.dot(dW2)
        dW1 = self.A[0].T.dot(dW1)


        # b1 -= db1 * learning_rate
        # W2 -= dW2 * learning_rate
        # b2 -= db2 * learning_rate
        # W3 -= dW3 * learning_rate
        # b3 -= db3 * learning_rate

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
        y = 0

        momentum = {
            "W1": 0,
            "b1": 0,
            "W2": 0,
            "b2": 0,
            "W3": 0,
            "b3": 0
        }
        error_increase_tol = 10^10
        prev_delta = -1
        average_delta = 0

        for n in range(max_iter):
            prev_delta = average_delta

            y = self.forward(inputs, targets)

            error = y - targets

            updates = self.backward(error, learning_rate)


            average_delta = 0

            for var_str, delta in updates.items():
                update = (learning_rate * delta)
                self.var[var_str] -= update
                # momentum[var_str] = delta
                average_delta += np.mean(delta)

            average_delta = average_delta/6

            if(average_delta != -1):
                if(average_delta < prev_delta):
                    learning_rate *= 1.02
                elif((math.fabs(prev_delta - average_delta))) < error_increase_tol:
                    learning_rate /= 2

            grads.append(sum(error)/len(inputs))


        return grads, y