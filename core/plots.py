from BetterNeuralNetwork import BetterNeuralNetwork
from NeuralNetwork import NeuralNetwork
import utils
import matplotlib.pyplot as plt
import numpy as np
import activation as act
import skeleton as sk
import MSE
import time

BASE_PATH = '/Users/vaevictis/Documents/As1/docs/images'

learning_rates = [0.5,0.1,0.01,0.001,0.0001]

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

        errors = np.mean(np.array(errors).reshape(-1, 10), 1)
        grads = np.mean(np.array(grads).reshape(-1, 10), 1)
        grads2 = np.mean(np.array(grads2).reshape(-1, 10), 1)
        plt.title('eta={}'.format(eta))
        plt.plot(grads, label="normal".format(eta))
        plt.plot(grads2, label="adagrad".format(eta))
        plt.ylabel('grads')
        plt.xlabel('iterations')
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

        errors = np.mean(np.array(errors).reshape(-1, 10), 1)
        grads = np.mean(np.array(grads).reshape(-1, 10), 1)
        grads2 = np.mean(np.array(grads2).reshape(-1, 10), 1)
        plt.title('eta={}'.format(eta))
        plt.plot(grads, label="normal".format(eta))
        plt.plot(grads2, label="momentum".format(eta))
        plt.ylabel('grads')
        plt.xlabel('iterations')

        plt.legend()
        fig.savefig(BASE_PATH + '/momentum/momentum_plot_{}'.format(i))
        fig.clf()


def nn_vs_bnn():
    trainX, trainT = sk.twospirals(250, noise=0.6, twist=800)

    train_X, train_T, testX, testT = utils.get_train_and_test_data(trainX, trainT)
    seed = 0
    # seed = int(time.time())
    np.random.seed(seed)

    # np.random.seed(seed)
    model = BetterNeuralNetwork(True)
    # create layers
    model.add_input_layer(2, 30, act.relu, act.drelu)
    model.add_hidden_layer(20, act.relu, act.drelu)
    model.add_hidden_layer(15, act.relu, act.drelu)
    model.add_hidden_layer(10, act.relu, act.drelu)
    model.add_output_layer(1, act.tanh, act.dtanh)

    # model.load('test')

    y, grads, errors, accuracy, accuracy_val = model.train(train_X, train_T, 8000, { 'eta' : 0.1, 'beta' : 0.5 }, 'adagrad',testX, testT)
    # model.save('test')
    errors = np.mean(np.array(errors).reshape(-1, 10), 1)
    grads = np.mean(np.array(grads).reshape(-1, 10), 1)
    accuracy = np.mean(np.array(accuracy).reshape(-1, 10), 1)
    accuracy_val = np.mean(np.array(accuracy_val).reshape(-1, 10), 1)

    np.random.seed(seed)
    nn = BetterNeuralNetwork(True)
    # # create layers
    nn.add_input_layer(2, 20, act.tanh, act.dtanh)
    nn.add_hidden_layer(15, act.tanh, act.dtanh)
    nn.add_output_layer(1)

    y_nn, grads_nn, errors_nn, accuracy_nn, accuracy_val_nn = nn.train(train_X, train_T, 8000, { 'eta' : 0.01, 'beta' : 0.5 }, 'gradient_descent',testX, testT)

    errors_nn = np.mean(np.array(errors_nn).reshape(-1, 10), 1)
    grads_nn = np.mean(np.array(grads_nn).reshape(-1, 10), 1)
    accuracy_nn = np.mean(np.array(accuracy_nn).reshape(-1, 10), 1)
    accuracy_val_nn = np.mean(np.array(accuracy_val_nn).reshape(-1, 10), 1)

    acc_train = sk.compute_accuracy(model, train_X, train_T)
    acc_test = sk.compute_accuracy(model, train_X, train_T)

    print("Accuracy from scratch Train: ", acc_train)
    print("Accuracy from scratch Test: ", acc_test)

    fig = plt.figure()
    plt.title('BNN vs NN: grads')
    plt.plot(grads, label="BNN, eta=0.1")
    plt.plot(grads_nn, label="NN, eta=0.1")
    plt.legend()
    fig.savefig('/Users/vaevictis/Documents/As1/docs/images/BNN_vs_NN/BNN_vs_NN_grads.png')


    fig = plt.figure()
    plt.title('BNN vs NN: error')
    plt.plot(errors, label="BNN, eta=0.1")
    plt.plot(errors_nn, label="NN, eta=0.1")
    plt.legend()

    fig.savefig('/Users/vaevictis/Documents/As1/docs/images/BNN_vs_NN/BNN_vs_NN_error.png')

    fig = plt.figure()
    plt.title('BNN vs NN: accuracy')
    plt.plot(accuracy, label="BNN, eta=0.1")
    plt.plot(accuracy_nn, label="NN, eta=0.1")
    plt.legend()
    fig.savefig('/Users/vaevictis/Documents/As1/docs/images/BNN_vs_NN/BNN_vs_NN_accuracy.png')

    # plt.show()

    fig = plt.figure()
    plt.title('BNN vs NN: test accuracy')
    plt.plot(accuracy_val, label="BNN, eta=0.1")
    plt.plot(accuracy_val_nn, label="NN, eta=0.1")
    plt.legend()
    fig.savefig('/Users/vaevictis/Documents/As1/docs/images/BNN_vs_NN/BNN_vs_NN_test_accuracy.png')

    fig = plt.figure()
    plt.subplot(211)

    plt.title("BNN: train={}, test={}".format(acc_train, acc_test))
    sk.plot_boundary(model,train_X,train_T,0.5)
    #
    acc_train = sk.compute_accuracy(nn, train_X, train_T)
    acc_test = sk.compute_accuracy(nn, train_X, train_T)

    plt.subplot(212)
    plt.title("NN: train={}, test={}".format(acc_train, acc_test))
    sk.plot_boundary(nn,train_X,train_T,0.5)
    plt.tight_layout()
    # plt.show()
    #
    print("Accuracy from scratch Train: ", acc_train)
    print("Accuracy from scratch Test: ", acc_test)

    # fig = plt.figure()
    # plt.title("train={0:.3f}, test={1:.3f}".format(acc_train, acc_test))
    # sk.plot_boundary(model,train_X,train_T,0.5)

    # plt.show()
    # fig.savefig('/Users/vaevictis/Documents/As1/docs/images/competition/competition_{}.png'.format(seed))
    fig.savefig('/Users/vaevictis/Documents/As1/docs/images/BNN_vs_NN/BNN_vs_NN_boundary.png')

    # fig.clf()

def cost_vs_eta():
    train_X, train_T = sk.twospirals()

    fig = plt.figure()
    for eta in learning_rates:
        model = create_normal_model()
        params = { 'eta' : eta}
        y, grads, errors, accuracy, accuracy_val = model.train(train_X, train_T, 10000, params)
        errors = np.mean(np.array(errors).reshape(-1, 10), 1)
        plt.plot(errors,label='eta={}'.format(eta))
    plt.ylabel('error')
    plt.xlabel('iterations')
    plt.legend()
    fig.savefig(BASE_PATH + '/NN_boundary_vs_learning_rates/errors.png')


def grads_vs_eta():
    train_X, train_T = sk.twospirals()

    fig = plt.figure()
    for eta in learning_rates:
        model = create_normal_model()
        params = { 'eta' : eta}
        y, grads, errors, accuracy, accuracy_val = model.train(train_X, train_T, 10000, params)
        grads = np.mean(np.array(grads).reshape(-1, 10), 1)
        plt.plot(grads,label='eta={}'.format(eta))
    plt.ylabel('grads')
    plt.xlabel('iterations')
    plt.legend()
    fig.savefig(BASE_PATH + '/NN_boundary_vs_learning_rates/grads.png')


def plot_part_2():
    train_X, train_T = sk.twospirals()
    for i in range(len(learning_rates)):
        model = create_normal_model()
        eta = learning_rates[i]

        y, grads, errors, accuracy, accuracy_val  = model.train(train_X,train_T,40000,{'eta':eta})
        acc_train = sk.compute_accuracy(model, train_X, train_T)
        fig = plt.figure()
        plt.title("eta={0}, accuracy={1:0.3f}, errors={2:0.3f}".format(eta,acc_train,MSE.MSE(y,train_T)))
        sk.plot_boundary(model,train_X,train_T,0.5)
        print("Accuracy from scratch Train: ", acc_train)
        fig.savefig(BASE_PATH + '/NN_boundary_vs_learning_rates/{}.png'.format(i))
        # fig.clf()


# gradient_desc_vs_momentum()
# gradient_desc_vs_adagrad()
# nn_vs_bnn()
plot_part_2()
# grads_vs_eta()
# cost_vs_eta()