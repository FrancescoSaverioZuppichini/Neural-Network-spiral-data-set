from Perceptron import Perceptron
from NeuralNetwork import NeuralNetwork as NN
from BetterNeuralNetwork import BetterNeuralNetwork as BNN
import skeleton as sk
import activation as act

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
    print('All threads finish after {:0.3f} ms'.format(float(time2-time1)*1000.0))
    return master




LEARNING_RATE = 0.001
MAX_ITER = 6000
STEP = 1

train_size = 200

X_train,T_train = sk.twospirals()

grads = []

for i in range(0):
    print('****************************')
    BNN_acc, NN_acc = sk.competition_train_from_scratch(X_train,T_train)
#
#     BNN_accs.append(BNN_acc)
#     NN_accs.append(NN_acc)
#
# plt.plot(BNN_accs,label='BNN')
# plt.plot(NN_accs,label='NN')
# plt.legend()
# plt.show()

# learning_rates = [0.05,0.01,0.005,0.001]

learning_rates = [0.001]

size = 1100
X, T = sk.twospirals(600)
X_train, T_train = X[:size],T[:size]
X_test, T_test = X[size:],T[size:]

# np.random.seed(1)
ITER = 4000
for i in range(1):
    for eta in learning_rates:
        # seed = int(time.time())
        # np.random.seed(seed)

        np.random.seed(i)

        # nn = NN()
        bnn = BNN()
        bnn.addInputLayer(2, 20, np.tanh, act.dtanh)
        bnn.addHiddenLayer(15, np.tanh, act.dtanh)
        bnn.addOutputLayer(1)

        # np.random.seed(seed)
        np.random.seed(i)

        #
        bnn2 = BNN()
        bnn2.addInputLayer(2, 20)
        bnn2.addHiddenLayer(15, act.tanh, act.dtanh)
        bnn2.addHiddenLayer(15, act.tanh, act.dtanh)
        # bnn2.addHiddenLayer(8, act.tanh, act.dtanh)
        bnn2.addOutputLayer(1)

        y, grads, BNN_errros = bnn2.train(X_train,T_train,eta,ITER)
        y, grads, NN_errros = bnn.train(X_train,T_train,eta,ITER)
        # y, grads, BNN_errros = bnn2.train(X_train,T_train,eta,ITER)

        # testX, testT = sk.twospirals(50)

        # np.random.seed(i)
        # net = NN()
        # net.train(X_train,T_train,eta,ITER)
        #
        # print("Accuracy from scratch NN: ", sk.compute_accuracy(net,X_test,T_test))

        print("Accuracy from scratch BNN: ", sk.compute_accuracy(bnn,X_test,T_test))

        print("Accuracy from scratch BNN with momentum: ", sk.compute_accuracy(bnn2,X_test,T_test))

        # plt.title("BNN vs NN with eta={}".format(eta))
        # NN_errros = np.mean(np.array(NN_errros).reshape(-1, 100), 1)
        # BNN_errros = np.mean(np.array(BNN_errros).reshape(-1, 100), 1)
        #
        # fig = plt.figure()
        # plt.plot(NN_errros,label='normal')
        # plt.plot(BNN_errros,label='momentum')
        # plt.legend()
        # plt.show()
        # fig.savefig('/Users/vaevictis/Documents/As1/docs/images/momentum_plot_{}.png'.format(eta))




X,T = sk.twospirals(120)
X_train = X[0:train_size]
T_train = T[0:train_size]
X_test = X[train_size:]
T_test = T[train_size:]
for i in range(0):
    np.random.seed(int(time.time()))
    # np.random.seed(1)
    #
    # net = NN()
    #
    net = BNN()
    net.addInputLayer(2, 20, act.tanh, act.dtanh)
    net.addHiddenLayer(15, act.tanh, act.dtanh)
    net.addOutputLayer(1)

    y,grads = net.train(X_train,T_train, LEARNING_RATE , MAX_ITER)
    # plt.plot(grads)
    # plt.show()
    print('--------')
    print('TRAINING SET')
    print(np.mean(((y > 0.5) * 1 == T_train) * 1))

    print('--------')
    print('TEST SET')
    y = net.forward(X_test)
    print(np.mean(((y > 0.5) * 1 == T_test) * 1))




# grads = [grads[i] for i in range(len(grads)) if i % STEP == 0]
# plt.plot(grads)
# plt.show()
# print(error)
#
# plt.figure(1)
#
# plt.subplot(211)
# plt.title('costs')
# plt.plot(costs)
#
# plt.subplot(212)
#
# plt.title('grad')
# plt.plot(grads)
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