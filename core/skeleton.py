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
from utils import get_train_and_test_data

from Perceptron import Perceptron
from NeuralNetwork import NeuralNetwork
from BetterNeuralNetwork import BetterNeuralNetwork
import activation as act
from MSE import MSE
import time
from MSE import dMSE

## Part 1

def get_part1_data():
    """
    Returns the toy data for the first part.
    """
    X = np.array([[1, 8],[6,2],[3,6],[4,4],[3,1],[1, 6],
              [6,10],[7,7],[6,11],[10,5],[4,11]])
    T = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]).reshape(-1, 1)
    return X, T

def train_one_step(model, learning_rate, inputs, targets):
    """
    Uses the forward and backward function of a model to compute the error and updates the model
    weights while overwritting model.var. Returns the cost.
    """
    y = model.forward(inputs)

    error = dMSE(y,targets)

    updates = model.backward(error)

    for var_str, delta in updates.items():
        update = delta * learning_rate
        model.var[var_str] -= update

    return y

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
    MAX_ITER = 15

    model = Perceptron()

    X, T = get_part1_data()

    y = None

    learning_rate = 0.02

    for n in range(MAX_ITER):
        y = train_one_step(model, learning_rate, X, T)

    plot_boundary(model, X, T)
    plt.savefig('/Users/vaevictis/Documents/As1/docs/images/run_part1.png')


    return y


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


def competition_train_from_scratch(testX, testT):
    """
    Trains the BetterNeuralNet model from scratch using the twospirals data and calls the other
    competition funciton to check the accuracy.
    """
    trainX, trainT = twospirals(250, noise=0.6, twist=800)
    train_X = trainX
    train_T = trainT
    # train_X, train_T, testX, testT = get_train_and_test_data(trainX, trainT)
    # seed = int(time.time())
    np.random.seed(1508171833)
    # np.random.seed(10)
    model = BetterNeuralNetwork(True)
    model.add_input_layer(2, 20, act.tanh, act.dtanh)
    model.add_hidden_layer(15, act.tanh, act.dtanh)
    model.add_hidden_layer(15, act.tanh, act.dtanh)
    model.add_output_layer(1)

    y, grads, errors, accuracy, accuracy_val = model.train(train_X, train_T ,8000, { 'eta' : 0.1, 'beta' : 0.5 },'adagrad', testX, testT)

    acc_train = compute_accuracy(model, trainX, trainT)
    acc_test = compute_accuracy(model, testX, testT)

    # plt.title("train={}, test={}".format(acc_train, acc_test))
    # plot_boundary(model,trainX,trainT,0.5)
    # plt.show()

    print("Accuracy from scratch Train: ", acc_train)
    print("Accuracy from scratch Test: ", acc_test)

    fig = plt.figure()
    plt.title("train={0:.3f}, test={1:.3f}".format(acc_train, acc_test))
    plot_boundary(model,trainX,trainT,0.5)

    plt.show()
    # fig.savefig('/Users/vaevictis/Documents/As1/docs/images/competition/competition_{}.png'.format(seed))

    return model


def competition_load_weights_and_evaluate_X_and_T(testX, testT):
    """
    Loads the weight values from a file into the BetterNeuralNetwork class and computes the accuracy.
    """
    NN = BetterNeuralNetwork()

    ## Implement



    ## End

    print("Accuracy from trained model: ", compute_accuracy(NN, testX, testT))




