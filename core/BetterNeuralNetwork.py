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

        self.W = np.random.randn(size_in,size_out)/np.sqrt(size_in)
        self.b = np.random.randn(1,size_out)

        self.dW = [np.zeros(self.W.shape)]
        self.db = [np.zeros([1,1])]
        # used for adagrad
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

    def gradient_descent(self, l, params):
        """
        Vanilla gradient descent.
        """
        learning_rate = params['eta']

        update_W = learning_rate * l.dW[1]
        update_b = learning_rate * l.db[1]

        return update_W, update_b

    def momentum(self, l, params):
        """
        Momentum implementation. More detail can be found here:
        http://ruder.io/optimizing-gradient-descent/#momentum
        """
        learning_rate = params['eta']

        beta = params['beta']

        update_W = learning_rate * l.dW[1] + beta * l.dW[0]
        update_b = learning_rate * l.db[1] + beta * l.db[0]

        return update_W, update_b

    def adagrad(self, l, params):
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


    def update_layers(self,params, method):
        """
        Update each layer sequentially calling the correct gradient descent method
        """
        for l in self.layers:

            update_W, update_b = method(l,params)

            l.W -= update_W
            l.b -= update_b

            l.dW = [update_W]
            l.db = [update_b]

    def forward(self, inputs):
        """
        Compute forward and store booth activations and weighted outputs.
        """
        self.A = [inputs]
        self.Z = []

        a = inputs

        for l in self.layers:
            z = a.dot(l.W) + l.b
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

        # start from the last layer
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
            db = np.sum(dW, axis=0, keepdims=True)

            l.dW.append(dW)
            l.db.append(db)

            i -= 1

    def one_train_step(self, x, t, params, method):
        """
        Compute one forward, one backpropagation and one weights update
        """
        y = self.forward(x)

        dx = (y - t) / len(x)

        self.backward(dx)

        self.update_layers(params, method)

        return y, dx

    def compute_batches(self, inputs, targets, params):
        X = [inputs]
        T = [targets]

        # set default size to 1
        params.setdefault('batch_size', 1)

        batch_size = params['batch_size']

        if (batch_size > 1):
            step = len(inputs) / batch_size

            for i in range(batch_size):
                X.append(inputs[int(step * i): int(step * (i + 1))])
                T.append(targets[int(step * i): int(step * (i + 1))])

        return X, T

    @timing
    def train(self,inputs, targets, max_iter, params=None, type='gradient_descent', X_val=[],T_val=[]):
        """
        Train the Neural Network with the given parameters

        :param inputs: The inputs vector
        :param targets: The targets vector
        :param max_iter:  Number of maximum iterations
        :param params: The parameters
        :param type: The name of the gradient descent method to use
        :param X_val: Validation inputs
        :param T_val:  Validation targets
        :return:
        """

        if(params == None):
            params = { 'eta':0.001 }

        grads = []
        errors = []
        accuracy = []
        accuracy_val = []

        y = 0

        method = self.update_func[type]

        X, T = self.compute_batches(inputs, targets, params)

        for n in range(max_iter):
            for i in range(len(X)):

                x = X[i]
                t = T[i]

                y, dx = self.one_train_step(x,t,params,method)

                if (self.DEBUG):
                    errors.append(cost_func.MSE(y, t))
                    grads.append(np.mean(np.abs(dx)))
                    acc = np.mean(((y > 0.5) * 1 == T) * 1)
                    accuracy.append(acc)

                    if (len(X_val) > 0):
                        acc = np.mean(((self.forward(X_val) > 0.5) * 1 == T_val) * 1)
                        accuracy_val.append(acc)

            #
            # if(n % 200 == 0):
            #     plt.title("acc={0:0.3f}, test={1:0.3f},iter={2:0.3f}, err={3:0.3f}".format(accuracy[-1], accuracy_val[-1],n, errors[-1]))
            #     plot_boundary(self, inputs, targets,0.5)
            #     plt.show(block=False)
            #     plt.pause(0.001)
            #     plt.clf()

        return y, grads, errors, accuracy, accuracy_val
    def save(self,file_name):
        """
        Save the current status into a file.
        """
        stuff = { 'size' : len(self.layers) }

        for i in range(len(self.layers)):
            l = self.layers[i]

            stuff['W_{}'.format(i)] = l.W
            stuff['b_{}'.format(i)] = l.b

        np.save(file_name,stuff)

        print("******* saved into {} *******".format(file_name))


    def load(self,file_name):
        """
        Load the current status from a file.
        """
        # TODO add file check
        stuff = np.load(file_name + ".npy").item()
        # unlock net
        self.freeze = False

        for i in range(stuff['size']):

            self.layers[i].W = stuff['W_{}'.format(i)]
            self.layers[i].b = stuff['b_{}'.format(i)]

        # lock net
        self.freeze = True

