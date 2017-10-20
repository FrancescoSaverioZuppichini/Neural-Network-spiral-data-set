# USI - Università della svizzera italiana
#
# Machine Learning - Fall 2017
#
# Assignment 1: Neural Networks
# Code Skeleton

import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron
from NeuralNetwork import NeuralNetwork
from BetterNeuralNetwork import BetterNeuralNetwork
import activation_function as act
from plot_boundary import plot_boundary
from cost_functions import MSE
from cost_functions import dMSE

import time

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
    plt.show()
    # plt.savefig('/Users/vaevictis/Documents/As1/docs/images/run_part1.png')


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
        y1 = NN.forward(X)
        error = (y1-T) / len(X)
        updates = NN.backward(error)
        print(type(updates[key]))
        if(len(updates[key].shape)==0):
            analytic_grad = updates[key]
        else:
            analytic_grad = updates[key][row, col]
        x1 = MSE(y1, T)

        NN.var[key][row][col] += eps
        y2 = NN.forward(X)
        # updates = NN.backward(y2 - T)
        x2 = MSE(y2, T)

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

    X,T = twospirals()

    np.random.seed(0)

    model = NeuralNetwork()

    res = model.train(X, T, 0.1, 40000)

    plot_boundary(model,X,T,0.5)

    # plt.show()

    print('Error: {}'.format(MSE(res[0],T)))
    print('Accuracy: {}'.format(compute_accuracy(model, X, T)))



def competition_train_from_scratch(testX, testT):
    """
    Trains the BetterNeuralNet model from scratch using the twospirals data and calls the other
    competition funciton to check the accuracy.
    """
    train_X, train_T = twospirals(250, noise=0.6, twist=800)

    # train_X, train_T, testX, testT = get_train_and_test_data(train_X,train_T,80)
    seed = int(time.time())
    # np.random.seed(1508255316)
    #
    seed = 0
    np.random.seed(seed)
    model = BetterNeuralNetwork(animation=True)
    # create layers
    model.add_input_layer(2, 30, act.relu, act.drelu)
    model.add_hidden_layer(20, act.relu, act.drelu)
    model.add_hidden_layer(15, act.relu, act.drelu)
    model.add_hidden_layer(10, act.relu, act.drelu)
    model.add_output_layer(1, act.tanh, act.dtanh)


    res = model.train(train_X, train_T, 4000, { 'eta' : 0.1}, 'adagrad',testX, testT)

    # model.save('competition')

    acc_train = compute_accuracy(model, train_X, train_T)
    acc_test = compute_accuracy(model, testX, testT)
    print("Accuracy from scratch Train: ", acc_train)
    print("Accuracy from scratch Test: ", acc_test)

    plt.title("train={0:.3f}, test={1:.3f}".format(acc_train, acc_test))
    plot_boundary(model, train_X, train_T, 0.5)
    plt.show()
    return model


def competition_load_weights_and_evaluate_X_and_T(testX, testT):
    """
    Loads the weight values from a file into the BetterNeuralNetwork class and computes the accuracy.
    """
    model = BetterNeuralNetwork()

    model.add_input_layer(2, 30, act.relu, act.drelu)
    model.add_hidden_layer(20, act.relu, act.drelu)
    model.add_hidden_layer(15, act.relu, act.drelu)
    model.add_hidden_layer(10, act.relu, act.drelu)
    model.add_output_layer(1, act.tanh, act.dtanh)

    model.load('competition')

    ## End

    print("Accuracy from trained model: ", compute_accuracy(model, testX, testT))




