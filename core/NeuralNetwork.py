import numpy as np
import activation as act
import MSE as cost_func

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
         "W1": np.random.random([L_sizes[0],2]),
         "b1": np.random.random([L_sizes[0],1]),
         "W2": np.random.random([L_sizes[1],L_sizes[0]]),
         "b2": np.random.random([L_sizes[1],1]),
         "W3": np.random.random([1,L_sizes[1]]),
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
        inputs =  np.array([inputs]).T
        # find out each z -> zl = W^l * a^{l-1} + b^l
        z1 = np.dot(W1,inputs) + b1
        a_1 = act.sigmoid(z1)

        z2 = np.dot(W2,a_1) + b2
        a_2 = act.tanh(z2)

        z3 =  np.dot(W3,a_2) + b3
        a_3 = act.tanh(z3)

        self.Z =  [z1,z2,z3]
        self.A = [inputs, a_1, a_2, a_3]

        # compute output error for each layer
        # delta^l = nabla_cost_l * act_func(z^l)
        for i in range(len(self.Z)):
            # skip first layer, they are the input
            nabla_c = cost_func.dMSE(self.A[i + 1],t)
            delta_act = act.dtanh(self.Z[i])
            delta = nabla_c * delta_act
            self.deltas.append(delta)

        y = z3

        ## End
        return y

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
        deltas = self.deltas

        delta = self.deltas[-1]
        grads = []

        dW3 = self.deltas[-1]
        db3 = dW3

        # dW3 = np.dot(dW3,self.A[2].T)

        dW2 = np.dot(W3.T, self.deltas[2]) * act.dtanh(Z[1])
        dW2 = np.dot(dW2,self.A[1].T)

        db2 = np.dot(b3.T,self.deltas[2]) * act.dtanh(Z[1])

        dW1 = np.dot(W2.T,self.deltas[1]) * act.dsigmoid(Z[0])
        dW1 = np.dot(dW1,self.A[0].T)
        db1 = np.dot(b2.T,self.deltas[1]) * act.dsigmoid(Z[0])


        ## End
        updates = {"W1": dW1,
                   "b1": db1,
                   "W2": dW2,
                   "b2": db2,
                   "W3": dW3,
                   "b3": db3}

        return updates


    def train(self,inputs,targets,learning_rate=0.1, max_iter=10):
        grads = []
        costs = []

        for n in range(max_iter):
            grad = 0
            cost = 0
            for i in range(len(inputs)):
                t = targets[i]
                y = self.forward(inputs[i],t)

                updates = self.backward(None)

                for var_str, delta in updates.items():

                    self.var[var_str] -= (learning_rate * delta)

                cost += cost_func.MSE(y,t)[0]
                grad += cost_func.dMSE(y,t)[0]

            grads.append(grad/len(inputs))
            costs.append(cost/len(inputs))

        return grads, costs