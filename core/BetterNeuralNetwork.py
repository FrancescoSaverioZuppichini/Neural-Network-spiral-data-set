import numpy as np
import activation as act
import MSE as cost_func
from utils import timing
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
class Layer:

    def __init__(self,size_in,size_out, activation=act.sigmoid, d_activation=act.dsigmoid):
        self.shape = [size_in,size_out]

        # according to this http://cs231n.github.io/neural-networks-2/
        # bias must be small, so let's scale them
        self.bias_scale = 0.01

        # self.gain = np.ones([size_out])

        self.W = np.random.randn(size_in,size_out)/np.sqrt(size_in)
        self.b = np.random.random([1,1]) * self.bias_scale

        self.dW = [np.zeros(self.W.shape)]
        self.db = [np.zeros([1,1])]

        # self.dW = 0
        # self.db = 0

        self.activation = activation
        self.d_activation = d_activation

class BetterNeuralNetwork:
    """
    Keeps track of the variables of the Multi Layer Perceptron model. Can be
    used for predictoin and to compute the gradients.
    """
    def __init__(self):

        # will hold all the intermediate quantity
        self.Z = []
        # will hold all the activation functions
        self.A = []
        # hold all the layers
        self.layers = []
        # freeze the network when the output layer as been added
        # or no input layer is added
        self.freeze = True



    def createLayer(self,size_in,size_out,activation,d_activation):

        if(self.freeze):
            return

        new_layer = Layer(size_in,size_out,activation,d_activation)
        return new_layer


    def addInputLayer(self,size_in,size_out,activation=act.sigmoid, d_activation=act.dsigmoid):
        self.freeze = False

        self.layers.append(self.createLayer(size_in,size_out,activation,d_activation))


    def addOutputLayer(self, size_out,activation=act.sigmoid, d_activation=act.dsigmoid):
        prev_layer_size = self.layers[-1].shape[1]

        self.layers.append(self.createLayer(prev_layer_size,size_out,activation,d_activation))

        self.freeze = True


    def addHiddenLayer(self,size, activation=act.sigmoid, d_activation=act.dsigmoid):
        prev_layer_size = self.layers[-1].shape[1]

        self.layers.append(self.createLayer(prev_layer_size,size,activation,d_activation))


    def forward(self, inputs):
        x = self.x = inputs

        self.A = [inputs]

        a = inputs

        for l in self.layers:
            z = a.dot(l.W) + l.b
            a = l.activation(z)
            # store
            self.A.append(a)
            self.Z.append(z)

        return self.A[-1]


    def backward(self, error):
        Z = self.Z
        A = self.A

        delta = error

        for i in range(1,len(self.layers)):
            l = self.layers[-i]

            dW  = delta * l.d_activation(Z[-i])
            db = np.mean(dW)

            delta = dW.dot(l.W.T)

            dW = A[-i -1].T.dot(dW)
            # directly update weight -> FASTER!
            # l.dW = dW
            # l.db = db
            l.db.append(db)
            l.dW.append(dW)
            # l.b -= db * learning_rate
            # l.W -= dW * learning_rate


    @timing
    def train(self,inputs,targets,learning_rate=0.001, max_iter=200):
        grads = []

        Beta = 0
        g = 0
        epoch = 1
        n = 1

        for n in range(max_iter):

            y = self.forward(inputs)

            error = y - targets

            self.backward(error)

            if(n % 100 == 1):
                print(np.mean(np.abs(error)))

            for i in range(1,len(self.layers)):
                l = self.layers[i]
                mom  = 0.5

                update_W = l.dW[1] * learning_rate + mom  * l.dW[0]
                update_b = l.db[1] * learning_rate + mom  * l.db[0]

                l.W -= update_W
                l.b -= update_b

                l.dW = [update_W]
                l.db = [update_b]




            # grads.append(np.sum(error)/len(inputs))


        return y, grads