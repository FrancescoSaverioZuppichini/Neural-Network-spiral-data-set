from BetterNeuralNetwork import BetterNeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
import activation as act
import skeleton as sk
import time

BASE_PATH = '/Users/vaevictis/Documents/As1/docs/images'

learning_rates = [0.1,0.01,0.001,0.0001]

X,T = sk.twospirals(250, noise=0.6, twist=800)

global_seed = int(time.time())

def create_normal_model():
    np.random.seed(global_seed)

    model = BetterNeuralNetwork(True)
    model.add_input_layer(2, 20, np.tanh, act.dtanh)
    model.add_hidden_layer(15, np.tanh, act.dtanh)
    model.add_output_layer(1)

    return model

def gradient_desc_vs_adagrad():
    fig = plt.figure()

    for i in range(len(learning_rates)):
        eta = learning_rates[i]

        model1 = create_normal_model()
        model2 = create_normal_model()

        params1 = {'eta':eta}
        params2 = {'eta':eta,'beta':0.5}

        y, grads, errors, accuracy, accuracy_val = model1.train(X, T, 10000, params1)
        y2, grads2, errors2, accuracy2, accuracy_val2  = model2.train(X, T, 10000, params2,'adagrad')

        errors = np.mean(np.array(errors).reshape(-1, 100), 1)
        grads = np.mean(np.array(grads).reshape(-1, 100), 1)

        grads2 = np.mean(np.array(grads2).reshape(-1, 100), 1)
        plt.title('eta={}'.format(eta))
        plt.plot(grads, label="normal".format(eta))
        plt.plot(grads2, label="adagrad".format(eta))

        plt.legend()
        fig.savefig(BASE_PATH + '/adagrad/adagrad_plot_{}'.format(i))
        fig.clf()


def gradient_desc_vs_momentum():
    fig = plt.figure()

    for i in range(len(learning_rates)):
        eta = learning_rates[i]

        model1 = create_normal_model()
        model2 = create_normal_model()

        params1 = {'eta':eta}
        params2 = {'eta':eta,'beta':0.5}

        y, grads, errors, accuracy, accuracy_val = model1.train(X, T, 10000, params1)
        y2, grads2, errors2, accuracy2, accuracy_val2 = model2.train(X, T, 10000, params2,'momentum')

        errors = np.mean(np.array(errors).reshape(-1, 100), 1)
        grads = np.mean(np.array(grads).reshape(-1, 100), 1)

        grads2 = np.mean(np.array(grads2).reshape(-1, 100), 1)
        plt.title('eta={}'.format(eta))
        plt.plot(grads, label="normal".format(eta))
        plt.plot(grads2, label="momentum".format(eta))

        plt.legend()
        fig.savefig(BASE_PATH + '/momentum/momentum_plot_{}'.format(i))
        fig.clf()


gradient_desc_vs_momentum()
gradient_desc_vs_adagrad()