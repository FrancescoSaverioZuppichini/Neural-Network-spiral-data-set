from Perceptron import Perceptron
from NeuralNetwork import NeuralNetwork as NN
from BetterNeuralNetwork import BetterNeuralNetwork as BNN
import utils
import skeleton as sk
import activation as act
import MSE as cost
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
from threading import Thread
from queue import Queue

from utils import timing


def full_training(model, learning_rate, inputs, targets, maxIter, momentum, beta, training_offset):
    grads_average = []
    errros_average = []

    total_grads = []
    total_erros = []
    total_results = []

    ## Implement
    for n in range(maxIter):

        results, errors, grads = sk.train_one_step(model, learning_rate, inputs, targets, momentum, beta, training_offset)

        total_results.append(results)
        total_erros.append(errors)
        total_grads.append(grads)

        grads_average.append(sum(grads)/len(inputs))
        errros_average.append(sum(errors)/len(inputs))

    return total_results,grads_average, errros_average, total_grads,total_erros

# sk.run_part1()

# print(model.forward(X))

def parall_train(X, T, learning_rate=0.001, max_iter=200, max_workers=1,steps=2):

    master = NN()
    time1 = time.time()

    step = len(X) // max_workers
    # my_queue = Queue()

    for i in range(steps):
        threads = []
        slaves = []
        for i in range(max_workers):
            x_batch = X[(i) * step:(i + 1) * step]
            t_batch = T[(i) * step:(i + 1) * step]
            slave = NN()
            slaves.append(slave)
            t = Thread(target=slave.train,args=(x_batch, t_batch, learning_rate, max_iter))
            # my_queue.put(slave.train(x_batch, t_batch, learning_rate, max_iter))
            threads.append(t)

        print('Created {}'.format(len(threads)))
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        for var_str, var in master.var.items():
            slave_vars = [slave.var[var_str] for slave in slaves]

            mean = np.average(slave_vars,axis=0)
            master.var[var_str] = mean

    # results = [my_queue.get() for x in range(max_workers)]
    time2 = time.time()
    total_time = float(time2-time1)*1000.0
    print('All threads finish after {:0.3f} ms'.format(total_time))
    print('{} iterations per seconds'.format((max_iter * max_workers *steps)/ total_time))
    return master

# learning_rates = [0.05,0.01,0.005,0.001]

# sk.run_part1()

# X,T = sk.get_part1_data()
# p = Perceptron()
# sk.train_one_step(p,0.02,X,T)
# print(p.var['b'])
# sk.run_part1()
# y = sk.train_one_step(p,0.001,X,T)

# print(y)
# print(cost.MSE(y,T))
# print(np.mean(np.abs(y - T)))

learning_rates = [0.1,0.01,0.001,0.0001]

X,T = sk.twospirals(300, noise=0.6, twist=800)
# X,T = sk.twospirals()
# X_train = X
# T_train = T
X_train, T_train, X_test, T_test  = utils.get_train_and_test_data(X,T)
# Dio, Cane, X_test, T_test  = utils.get_train_and_test_data(X,T)

# fig =  plt.figure()
# plt.title('Train')
# sk.plot_data(X_train,T_train)
# fig.savefig('/Users/vaevictis/Documents/As1/docs/images/train_set.png')
# fig2 =  plt.figure()
# plt.title('Test')
# sk.plot_data(X_test,T_test)
# fig2.savefig('/Users/vaevictis/Documents/As1/docs/images/test_set.png')

for i in range(1):
    sk.competition_train_from_scratch(X_test,T_test)
# seed = int(time.time())
# print(seed)
# np.random.seed(seed)

# bnn = BNN(True)
# bnn.add_input_layer(2, 20, np.tanh, act.dtanh)
# bnn.add_hidden_layer(15, np.tanh, act.dtanh)
# bnn.add_output_layer(1)

# bnn.load('test')
# print(len(X_train))

# fig = plt.figure()
for i in range(0):
    # seed = i
    # seed = int(time.time())
    np.random.seed(1508166319)
    # nn = NN()
    # nn.train(X_train,T_train,0.01,30000)
    bnn = BNN(True)
    bnn.add_input_layer(2, 20, np.tanh, act.dtanh)
    bnn.add_hidden_layer(15, np.tanh, act.dtanh)
    bnn.add_hidden_layer(15, np.tanh, act.dtanh)
    # bnn.add_hidden_layer(8, np.tanh, act.dtanh)
    bnn.add_output_layer(1)
    #
    params = {'eta':0.2,'beta':0.5}
    # # bnn = NN()
    # # bnn.train(X_train, T_train, 0.001, 3000)
    y, grads, errors, accuracy, accuracy_val = bnn.train(X_train, T_train, 4000, params, 'adagrad',X_test,T_test)
    # errors = np.mean(np.array(errors).reshape(-1, 10), 1)
    # grads = np.mean(np.array(grads).reshape(-1, 10), 1)
    # accuracy = np.mean(np.array(accuracy).reshape(-1, 10), 1)
    # accuracy_val = np.mean(np.array(accuracy_val).reshape(-1, 10), 1)

    # fig = plt.figure()
    sk.plot_boundary(bnn,X,T,0.5)
    plt.plot()
    # print(errors[-1])
    # plt.title("train={0:.3f}, seed={1}".format(accuracy[-1],seed))

    # plt.title("train={0:.3f}, test={0:.3f}".format(accuracy[-1], accuracy_val[-1]))
    # plt.plot(grads,label="grad")
    # plt.plot(errors, label='error')
    # plt.legend()
    # plt.show()
    # fig.savefig('/Users/vaevictis/Documents/As1/docs/images/competition/competition_{}.png'.format(seed))

    # plt.title("train={}, test={}".format(accuracy[-1], accuracy_val[-1]))
    # plt.plot(accuracy, label='train')
    # plt.plot(accuracy_val, label='validation')
    # plt.legend()
    # plt.show()

    # y = bnn.forward(X_train)
    # plt.title("Accuracy={}".format(sk.compute_accuracy(bnn,X_train,T_train)))
    # sk.plot_boundary(bnn,X_train,T_train,0.5)
    # plt.show()
    # plt.plot(lrs)
    # plt.show()

    print(sk.compute_accuracy(bnn,X_train,T_train))
    # print(sk.compute_accuracy(bnn,X_test,T_test))
# part 4
ITER = 10000

# np.random.seed(1)
for i in range(0):

    fig = plt.figure()
    seed = int(time.time())
    # print(seed)
    for eta in learning_rates:

        # seed = int(time.time())
        np.random.seed(seed)
        #
        # np.random.seed(10)
        #
        # nn = NN()
        bnn = BNN(True)
        bnn.add_input_layer(2, 20, np.tanh, act.dtanh)
        bnn.add_hidden_layer(15, np.tanh, act.dtanh)
        bnn.add_output_layer(1)

        # np.random.seed(seed)
        # np.random.seed(i)

        bnn2 = BNN(True)
        bnn2.add_input_layer(2, 20, np.tanh, act.dtanh)
        bnn2.add_hidden_layer(15, np.tanh, act.dtanh)
        bnn2.add_output_layer(1)

        y, grads, BNN_errros, accuracy, accuracy_val = bnn.train(X_train,T_train,ITER, {'eta':eta})
        y, grads_2, BNN_errros_2, accuracy, accuracy_val = bnn2.train(X_train,T_train,ITER,{'eta':eta,'beta':0.5},'adagrad')
        # y, grads, BNN_errros = bnn2.train(X_train,T_train,eta,ITER)

        # testX, testT = sk.twospirals(50)

        # np.random.seed(i)
        # net = NN()
        # net.train(X_train,T_train,eta,ITER)
        #
        print("Accuracy from scratch NN: ", sk.compute_accuracy(bnn,X_test,T_test))

        print("Accuracy from scratch BNN: ", sk.compute_accuracy(bnn2,X_test,T_test))

        # print("Accuracy from scratch BNN with momentum: ", sk.compute_accuracy(bnn2,X_test,T_test))

        # plt.title("BNN vs NN with eta={}".format(eta))
        BNN_errros = np.mean(np.array(grads).reshape(-1, 100), 1)
        BNN_errros_2 = np.mean(np.array(grads_2).reshape(-1, 100), 1)
        # fig = plt.figure()
        plt.title('Learning rates')
        #
        plt.title('Learning rate {}'.format(eta))
        plt.plot(BNN_errros,label='normal')
        plt.plot(BNN_errros_2,label='adagrad')
        plt.legend()
        # plt.show()
        fig.savefig('/Users/vaevictis/Documents/As1/docs/images/adagrad_plot_{}.png'.format(eta))
        plt.clf()
        #
        # plt.plot(BNN_errros_2,label='momentum')
    # plt.legend()
    # fig.savefig('/Users/vaevictis/Documents/As1/docs/images/NN_eta_vs_training_momentum.png')
    # plt.show()








DEBUG = False

# if(DEBUG):
    # fig = plt.figure()
    # ax = plt.axes(xlim=(0, 10), ylim=(-5, 5))
    # iter_text = ax.text(0.02, 0.95, '', )
    #
    # line, = ax.plot([], [], lw=2)
    #
    # # initialization function: plot the background of each frame
    # def init():
    #     line.set_data([], [])
    #     return line,
    #
    # # animation function.  This is called sequentially
    # def animate(i):
    #     x = np.arange(len(total_grads[i]))
    #     y = total_erros[i]
    #     print(i)
    #
    #     line.set_data(x, y)
    #
    #     return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    # anim = animation.FuncAnimation(fig, animate, init_func=init,frames=2000, interval=50, blit=True)
    # plt.show()
    # plt.plot(T, 'ro',label="T")
    #
    # lastResult = total_results[len(total_results) - 1]
    # plt.plot(lastResult, 'g',label="B")
    #
    #
    # # results = [results[i] for i in range(len(results)) if i % STEP == 0]
    # # errors = [errors[i] for i in range(len(errors)) if i % STEP == 0]
    #
    #
    # # plt.plot(lastResult, label='best')
    #
    # plt.legend()
    # plt.show()

    # plt.figure(1)
    #
    # plt.subplot(211)
    # plt.title('gradient')
    # plt.plot(grads_average, label='grad')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(212)
    # plt.title('error')
    # plt.plot(errros_average, label='errors')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.show()