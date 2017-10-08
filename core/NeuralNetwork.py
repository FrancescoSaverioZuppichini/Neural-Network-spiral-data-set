import numpy as np
import activation as act
import MSE as cost_func
from utils import timing

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

        self.var = {
         "W1": np.random.randn(2,20)/np.sqrt(2),
         "b1": np.random.random([1,1]),
         "W2": np.random.rand(20,15)/np.sqrt(20),
         "b2": np.random.random([1,1]),
         "W3": np.random.rand(15,1)/np.sqrt(15),
         "b3": np.random.random([1,1])
        }

        ## End

    def forward(self, inputs, t):
        """
        Implements the forward pass of the MLP model and returns the prediction y. We need to
        store the current input for the backward function.
        """
        x = self.x = inputs

        W1 = self.var['W1']
        b1 = self.var['b1']
        W2 = self.var['W2']
        b2 = self.var['b2']
        W3 = self.var['W3']
        b3 = self.var['b3']

        # prediction at each layer
        # find out each z -> zl = W^l * a^{l-1} + b^l
        z1 = inputs.dot(W1) + b1
        a_1 = np.tanh(z1)

        z2 = a_1.dot(W2) + b2
        a_2 = np.tanh(z2)

        z3 =  a_2.dot(W3) + b3
        a_3 = act.sigmoid(z3)

        self.Z = [z1,z2,z3]
        # self.A = [np.array([inputs]),a_1,a_2,a_3]

        self.A = [inputs,a_1,a_2,a_3]

        ## End
        return a_3

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

        # back propagate and compute each output error
        da3 = error * act.dsigmoid(Z[-1])
        dW3 = self.A[2].T.dot(da3)
        dh3 = da3.dot(W3.T)
        # print(A[2].shape,dw3.shape)
        db3 = np.mean(dh3)

        da2 = dh3 * act.dtanh(Z[1])
        dW2 = self.A[1].T.dot(da2)
        dh2 = da2.dot(W2.T)

        db2 = np.mean(dh2)

        da1 = dh2 * act.dtanh(Z[0])
        dW1 = self.A[0].T.dot(da1)
        dh1 = da1.dot(W1.T)

        db1 = np.mean(dh1)
        #
        # dW3 = error * act.dsigmoid(Z[-1])
        # db3 = np.mean(dW3)
        #
        #
        # dW2 = dW3.dot(W3.T) * act.dtanh(Z[1])
        # db2 = np.mean(dW2)
        #
        # dW1 = dW2.dot(W2.T) * act.dtanh(Z[0])
        # db1 = np.mean(dW1)
        #
        # # compute grads
        # dW3 = self.A[2].T.dot(dW3)
        # dW2 = self.A[1].T.dot(dW2)
        # dW1 = self.A[0].T.dot(dW1)

        ## End
        updates = {"W1": dW1,
                   "b1": db1,
                   "W2": dW2,
                   "b2": db2,
                   "W3": dW3,
                   "b3": db3}

        return updates

    @timing
    def train(self,inputs,targets,learning_rate=0.001, max_iter=3000):
        grads = []
        costs = []
        y = 0
        for n in range(max_iter):
            grad = 0
            cost = 0
            # for i in range(len(inputs)):
            #     y = self.forward(inputs[i],targets[i])
            #
            #     error = y - targets[i]
            #
            #     updates = self.backward(error)
            #     #
            #     for var_str, delta in updates.items():
            #     #
            #         self.var[var_str] -= learning_rate * delta
            #
            #     cost += cost_func.MSE(y, targets[i])
            #     grad += error[0]
            # #
            # costs.append(cost/len(inputs))
            # grads.append(grad/len(inputs))
            y = self.forward(inputs, targets)

            updates = self.backward(y - targets)

            for var_str, delta in updates.items():
                change = learning_rate * delta

                self.var[var_str] -= change
        print(np.mean(((y > 0.5) * 1 == targets) * 1))

        return grads, costs