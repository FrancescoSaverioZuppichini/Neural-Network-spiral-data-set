import numpy as np
import matplotlib.pyplot as plt
# custom imports
import activation as act
import MSE as cost_func
from utils import timing

def plot_data(X,T):
    """
    Plots the 2D data as a scatterplot
    """
    plt.scatter(X[:,0], X[:,1], s=40, c=T, cmap=plt.cm.Spectral)


def plot_boundary(model, X, targets, threshold=0.0):
    """
    Plots the data and the boundary lane which seperates the input space into two classes.
    """
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    y = model.forward(X_grid)
    plt.contourf(xx, yy, y.reshape(*xx.shape) < threshold, alpha=0.5)
    plot_data(X, targets)
    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])


class Layer:
    """
    Layer high-level implementation.
    """
    def __init__(self,size_in,size_out, activation=act.sigmoid, d_activation=act.dsigmoid):
        self.shape = [size_in,size_out]
        # according to this http://cs231n.github.io/neural-networks-2/
        # bias must be small, so let's scale them
        self.bias_scale = 0.01

        self.W = np.random.randn(size_in,size_out)/np.sqrt(size_in)
        self.b = np.random.randn(1,size_out) * self.bias_scale

        # self.dW = np.zeros(self.W.shape)
        # self.dB = np.zeros([1,1])
        self.dW = [np.zeros(self.W.shape)]
        self.db = [np.zeros([1,1])]
        # used for momentum
        self.cache = [0,0]

        self.activation = activation
        self.d_activation = d_activation

class BetterNeuralNetwork:
    """
    Keeps track of the variables of the Multi Layer Perceptron model. Can be
    used for predictoin and to compute the gradients.
    """
    def __init__(self, DEBUG=False):
        # will hold all the intermediate quantity
        self.Z = []
        # will hold all the activation functions
        self.A = []
        # hold all the layers
        self.layers = []
        # freeze the network when the output layer as been added
        # or no input layer is added
        self.freeze = True
        # enable debug flag to store useful data
        self.DEBUG = DEBUG
        # holds all available gradient descent update methods
        self.update_func = {
            'momentum': self.momentum,
            'adagrad': self.adagrad,
            'gradient_descent': self.gradient_descent }

    def create_layer(self,size_in,size_out,activation,d_activation):

        if(self.freeze):
            return

        new_layer = Layer(size_in,size_out,activation,d_activation)

        return new_layer


    def add_input_layer(self,size_in,size_out,activation=act.sigmoid, d_activation=act.dsigmoid):
        """
        Add a input layer, size_in is required.
        """
        self.freeze = False

        self.layers.append(self.create_layer(size_in,size_out,activation,d_activation))

    def add_output_layer(self, size_out,activation=act.sigmoid, d_activation=act.dsigmoid):
        """
        Add a output layer, size_in is calculated by the Network
        """
        prev_layer_size = self.layers[-1].shape[1]

        self.layers.append(self.create_layer(prev_layer_size,size_out,activation,d_activation))

        self.freeze = True

    def add_hidden_layer(self,size, activation=act.sigmoid, d_activation=act.dsigmoid):
        """
        Add a hidden layer, size_in is calculated by the Network
        """
        prev_layer_size = self.layers[-1].shape[1]

        self.layers.append(self.create_layer(prev_layer_size,size,activation,d_activation))


    def forward(self, inputs):
        """
        Compute forward and store booth activations and weighted outputs.
        """
        self.A = [inputs]
        self.Z = []
        # dropout probability
        # p = 0.2

        a = inputs

        for l in self.layers:
            z = a.dot(l.W) + l.b
            # create dropout mask
            # u = (np.random.rand(*z.shape) < p) / p
            # apply mask
            # z *= u
            a = l.activation(z)
            # store
            self.A.append(a)
            self.Z.append(z)

        return a


    def backward(self, error):
        """
        Compute backward propagation for each layer and store the result.
        """
        A = self.A
        Z = self.Z

        delta = error

        # backprop starting from the last layer
        i = len(self.layers) - 1
     
        while (i >= 0):
            l = self.layers[i]
            # d = (W^{l+1}).d^{l+1} * a^l
            dW = delta * l.d_activation(Z[i])
            # will be used next iteration
            delta = dW.dot(l.W.T)
            # get the gradient
            # grad = a^{l-1}.d^l
            dW = A[i].T.dot(dW)
            db = np.mean(dW)

            l.dW.append(dW)
            l.db.append(db)

            i -= 1

    def gradient_descent(self, l, params):
        """
        Vanilla gradient descent.
        """
        learning_rate = params['eta']

        update_W = learning_rate * l.dW[1]
        update_b = learning_rate * l.db[1]

        return update_W, update_b

    def momentum(self,l,params):
        """
        Momentum implementation. More detail can be found here:
        http://ruder.io/optimizing-gradient-descent/#momentum
        """
        learning_rate = params['eta']

        beta = params['beta']

        update_W = learning_rate * l.dW[1] + beta * l.dW[0]
        update_b = learning_rate * l.db[1] + beta * l.db[0]

        return update_W, update_b

    def adagrad(self, l,params):
        """
        Adagrad implementation. More detail can be found here:
        http://ruder.io/optimizing-gradient-descent/#adagrad
        """
        learning_rate = params['eta']
        eps = 1e-8

        l.cache[0] += l.dW[1] ** 2
        l.cache[1] += l.db[1] ** 2

        update_W = learning_rate * l.dW[1] / (np.sqrt(l.cache[0]) + eps)
        update_b = learning_rate * l.db[1] / (np.sqrt(l.cache[1]) + eps)

        return update_W, update_b


    def update_layers(self,params, type):
        """
        Update each layer sequentially calling the correct gradient descent method
        """
        for l in self.layers:

            update_W, update_b = self.update_func[type](l,params)

            l.W -= update_W
            l.b -= update_b

            l.dW = [update_W]
            l.db = [update_b]


    @timing
    def train(self,inputs, targets, max_iter, params={}, type='gradient_descent', batch_size=1, X_val=[],T_val=[]):
        """
        Train the Neural Network according to the given parameters

        Args:
            inputs : The inputs vector.
            targets: The targets vector.
            max_iter (int) : Number of maximum iterations.
            params (dict) : The parameters
            type (string) : The name of the gradient descent method to use
            X_val : Validation inputs.
            T_val : Validation targets.
        """
        grads = []
        errors = []
        accuracy = []
        accuracy_val = []

        y = 0

        X = [inputs]
        T = [targets]

        step = len(inputs) / batch_size

        for i in range(batch_size):
            X.append(inputs[int(step * i): int(step * (i + 1))])
            T.append(targets[int(step * i): int(step * (i + 1))])


        for n in range(max_iter):
            for i in range(len(X)):
                x = X[i]
                t = T[i]

                y = self.forward(x)

                dx = y - t

                self.backward(dx)

                self.update_layers(params,type)

                if (self.DEBUG):
                    errors.append(cost_func.MSE(y, t))
                    grads.append(np.mean(np.abs(dx)))
                    acc = np.mean(((y > 0.5) * 1 == T) * 1)
                    accuracy.append(acc)

                if (len(X_val) > 0):
                    acc = np.mean(((self.forward(X_val) > 0.5) * 1 == T_val) * 1)
                    accuracy_val.append(acc)

            # if(n % 100 == 0):
            #     plt.title("acc={}, iter={}, err={}".format(accuracy[-1],n, errors[-1]))
            #     plot_boundary(self, inputs, targets,0.5)
            #     plt.show(block=False)
            #     plt.pause(0.001)
            #     plt.clf()
        #
        return y, grads, errors, accuracy, accuracy_val

    def save(self,file_name):
        """
        Save the current status into a file.
        """
        # TODO Save activations

        W = []
        b = []

        for l in self.layers:
            W.append(l.W)
            b.append(l.b)

        np.save(file_name + "_W", W)
        np.save(file_name + "_b", b)

    def load(self,file_name):
        """
        Load the current status from a file.
        """
        # TODO add file check
        # TODO load activations
        W = np.load(file_name + "_W.npy")
        b = np.load(file_name + "_b.npy")

        # unlock net
        self.freeze = False

        for i in range(len(W)):
            j, k = W[i].shape

            # new_layer = self.create_layer(j,k,act.sigmoid,act.dsigmoid)

            self.layers[i].W = W[i]
            self.layers[i].b = b[i]

        # lock net
        self.freeze = True

