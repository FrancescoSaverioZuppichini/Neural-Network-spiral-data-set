#
# USI - Università della svizzera italiana
#
# Machine Learning - Fall 2017
#
# Assignment 1: Neural Networks
# Code Skeleton

import numpy as np
import math
import matplotlib.pyplot as plt
from utils import timing

## Part 1

def get_part1_data():
    """
    Returns the toy data for the first part.
    """
    X = np.array([[1, 8],[6,2],[3,6],[4,4],[3,1],[1, 6],
              [6,10],[7,7],[6,11],[10,5],[4,11]])
    T = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]).reshape(-1, 1)
    return X, T

def MSE(prediction, target):
    """
    Computes the Mean Squared Error of a prediction and its target
    """
    y = prediction
    t = target
    n = prediction.size

    if (y.size != n):
        raise Exception("Parameters must have the same len!")

    s = sum(y - t) ** 2

    meanCost = (0.5 / n) * s


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


    error = (y - t) * y*(1 - y)

    ## End
    return error

def sigmoid(x):

    return 1/(1 + math.e ** -x)

class Perceptron:
    """
    Keeps track of the variables of the Perceptron model. Can be used for predictoin and to compute the gradients.
    """
    def __init__(self):
        """
        The variables are stored inside a dictonary to make them easy accessible.
        """
        self.var = {
         "W": np.array([.8, -.5]),
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

        s = sum(x*W) + b
        y = sigmoid(s)

        ## End
        return y

    def backward(self, error):
        """
        Backpropagates through the model and computes the derivatives. The forward function must be
        run before hand for self.x to be defined. Returns the derivatives without applying them using
        a dictonary similar to self.var.
        """
        x = self.x

        ## Implement



        ## End
        updates = {"W": dW,
                   "b": db}

        return updates

@timing
def train_one_step(model, learning_rate, inputs, targets, maxIter):
    """
    Uses the forward and backward function of a model to compute the error and updates the model
    weights while overwritting model.var. Returns the cost.
    """
    costs = np.empty((0,len(inputs)))
    # results = np.empty((0,len(inputs)))


    results = []
    grads = []
    ## Implement
    for n in range(maxIter):
        gradValue = 0
        results.append([])
        # cost = np.array([])
        # result = np.array([])

        for i in range(len(inputs)):
            x = inputs[i]
            t = targets[i]

            y = model.forward(x)

            # result = np.append(result,np.array([y]))

            grad = dMSE(y, t)

            results[n].append(y)
            # cost = np.append(cost,np.array([grad]))

            model.var['W'] =  model.var['W'] - learning_rate * grad * x

            gradValue += grad

        gradValue /= len(inputs)

        grads.append(gradValue)
        # print(sum(cost)/len(inputs))
        # costs = np.vstack((costs,cost))
        # results = np.vstack((results, result))

    #for varstr, grad in updates.items():
    #    model.var[varstr] = (...)

    ## End
    return costs, results, grads

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

def run_part1():
    """
    Train the perceptron according to the assignment.
    """


## Part 2
def twospirals(n_points=120, noise=1.6, twist=420):
    """
     Returns a two spirals dataset.
    """
    np.random.seed(0)
    n = np.sqrt(np.random.rand(n_points,1)) * twist * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    X, T =(np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))
    T = np.reshape(T, (T.shape[0],1))
    return X, T

def compute_accuracy(model, X, T):
    """
    Computes the average accuracy over this data.
    """
    return np.mean(((model.forward(X) > 0.5)*1 == T)*1)



def dsigmoid(x):

    sigma = sigmoid(x)
    return sigma * (1.0 - sigma)

    ## End
    return x

def tanh(x):
    """
    Implements the hyperbolic tangent activation function.
    """
    ## Implement



    ## End
    return x

def dtanh(x):
    """
    Implements the derivative of the hyperbolic tangent activation function.
    """
    ## Implement



    ## End
    return x


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

        #W1_in = ...
        #W1_out = ...
        #W2_in = ...
        #W2_out = ...
        #W3_in = ...
        #W3_out = ...

        self.var = {
         #"W1": (...),
         #"b1": (...),
         #"W2": (...),
         #"b2": (...),
         #"W3": (...),
         #"b3": (...)
        }

        ## End

    def forward(self, inputs):
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

        ## Implement



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

        ## Implement



        ## End
        updates = {"W1": dW1,
                   "b1": db1,
                   "W2": dW2,
                   "b2": db2,
                   "W3": dW3,
                   "b3": db3}
        return updates

def gradient_check():
    """
    Computes the gradient numerically and analitically and compares them.
    """
    X, T = twospirals(n_points=10)
    NN = NeuralNetwork()
    eps = 0.0001

    for key,value in NN.var.items():
        row = np.random.randint(0,NN.var[key].shape[0])
        col = np.random.randint(0,NN.var[key].shape[1])
        print("Checking ", key, " at ",row,",",col)

        ## Implement
        #analytic_grad = ...

        #x1 =  ...
        NN.var[key][row][col] += eps
        #x2 =  ...

        ## End
        numeric_grad = (x2 - x1) / eps
        print("numeric grad: ", numeric_grad)
        print("analytic grad: ", analytic_grad)
        if abs(numeric_grad-analytic_grad) < 0.00001:
            print("[OK]")
        else:
            print("[FAIL]")

def run_part2():
    """
    Train the multi layer perceptron according to the assignment.
    """



## Part 3
class BetterNeuralNetwork:
    """
    Keeps track of the variables of the Multi Layer Perceptron model. Can be
    used for predictoin and to compute the gradients.
    """
    def __init__(self):
        """
        The variables are stored inside a dictonary to make them easy accessible.
        """
        ## Implement
        #W1_in = ...
        #W1_out = ...
        #W2_in = ...
        #W2_out = ...
        #W3_in = ...
        #W3_out = ...

        self.var = {
         #"W1": (...),
         #"b1": (...),
         #"W2": (...),
         #"b2": (...),
         #"W3": (...),
         #"b3": (...),
        }

        ## End

    def forward(self, inputs):
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

        ## Implement



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

        ## Implement



        ## End
        updates = {"W1": dW1,
                   "b1": db1,
                   "W2": dW2,
                   "b2": db2,
                   "W3": dW3,
                   "b3": db3}
        return updates

def competition_train_from_scratch(testX, testT):
    """
    Trains the BetterNeuralNet model from scratch using the twospirals data and calls the other
    competition funciton to check the accuracy.
    """
    trainX, trainT = twospirals(n_points=250, noise=0.6, twist=800)
    NN = BetterNeuralNetwork()

    ## Implement



    ## End

    print("Accuracy from scratch: ", compute_accuracy(NN, testX, testT))


def competition_load_weights_and_evaluate_X_and_T(testX, testT):
    """
    Loads the weight values from a file into the BetterNeuralNetwork class and computes the accuracy.
    """
    NN = BetterNeuralNetwork()

    ## Implement



    ## End

    print("Accuracy from trained model: ", compute_accuracy(NN, testX, testT))


X,T = get_part1_data()

MAX_ITER = 1000
STEP = 10
costs, results, grads = train_one_step(Perceptron(),0.5,X,T, MAX_ITER)

x = np.arange(T.size)

print(x)

# TODO utils function

plt.plot(x, T, 'ro',label="T")

lastResult = results[len(results) - 1]

plt.plot(x, lastResult, label='Y')

results = [results[i] for i in range(len(results)) if i % STEP == 0]

# for i in range(len(results)):
#     label = "%s" % (i)
#     line = plt.plot(x, results[i], label=label)

plt.legend()
plt.show()

# plt.plot(np.arange(maxIter), grads )
# plt.show()
