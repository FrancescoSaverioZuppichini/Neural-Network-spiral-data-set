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

learning_rates = [0.1,0.01,0.002,0.001,0.0001]

X_train, T_train, X_test, T_test = utils.get_data(data_func=sk.twospirals,train_ratio=80)


# sk.competition_train_from_scratch(X_test,T_test)
# master = parall_train(X_train,T_train,0.002,2000,4,4)
#
# print(sk.compute_accuracy(master,X_test,T_test))

# fig =  plt.figure()
# plt.title('Train')
# sk.plot_data(X_train,T_train)
# fig.savefig('/Users/vaevictis/Documents/As1/docs/images/train_set.png')
# fig2 =  plt.figure()
# plt.title('Test')
# sk.plot_data(X_test,T_test)
# fig.savefig('/Users/vaevictis/Documents/As1/docs/images/test_set.png')
# sk.plot_data(X_test,T_test)
#
# plt.show()
sk.competition_train_from_scratch(X_train,T_train)

# bnn = NN()
# X_train, T_train = sk.twospirals(250, noise=0.6, twist=800)

best = 100
seed_best = 0
for i in range(0):
    seed = i
    # seed = int(time.time())
    # print(seed)
    np.random.seed(seed)
    # np.random.seed(seed)
    bnn = BNN()
    bnn.addInputLayer(2, 20, np.tanh, act.dtanh)
    bnn.addHiddenLayer(15, np.tanh, act.dtanh)
    bnn.addOutputLayer(1)


    # np.random.seed(int(time.time()))
    y, grads, errors = bnn.train(X_train, T_train, 0.0002, 5000,True)
    errors = np.mean(np.array(errors).reshape(-1, 100), 1)
    grads = np.mean(np.array(grads).reshape(-1, 100), 1)
    print(errors[-1])
    plt.plot(grads,label="grad")
    plt.plot(errors, label='error')
    plt.legend()
    plt.show()
    # BNN_errros = np.mean(np.array(errors).reshape(-1, 100), 1)
    # plt.plot(BNN_errros)
    # y = bnn.forward(X_train)
    # y, grads, errors = bnn.train(X, T, 0.001, 5000)
    # sk.plot_boundary(bnn,X_train,T_train,0.5)
    # plt.show()
    # plt.plot(lrs)
    # plt.show()
    # cost_temp = cost.MSE(y,T_train)
    # if(cost_temp < best):
    #     best = cost_temp
    #     seed_best = seed
    #     print('** new found ***')
    #     print(seed_best)
    # print(cost_temp)

    print(sk.compute_accuracy(bnn,X_test,T_test))
print(seed_best)

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
        bnn = BNN()
        bnn.addInputLayer(2, 20, np.tanh, act.dtanh)
        bnn.addHiddenLayer(15, np.tanh, act.dtanh)
        bnn.addOutputLayer(1)

        # np.random.seed(seed)
        # np.random.seed(i)

        # bnn2 = BNN()
        # bnn2.addInputLayer(2, 20, np.tanh, act.dtanh)
        # bnn2.addHiddenLayer(15, np.tanh, act.dtanh)
        # bnn2.addOutputLayer(1)

        y, grads, BNN_errros = bnn.train(X_train,T_train,eta,ITER,True)
        # y, grads, BNN_errros_2 = bnn2.train(X_train,T_train,eta,ITER,True)
        # y, grads, BNN_errros = bnn2.train(X_train,T_train,eta,ITER)

        # testX, testT = sk.twospirals(50)

        # np.random.seed(i)
        # net = NN()
        # net.train(X_train,T_train,eta,ITER)
        #
        print("Accuracy from scratch NN: ", sk.compute_accuracy(bnn,X_test,T_test))

        # print("Accuracy from scratch BNN: ", sk.compute_accuracy(bnn2,X_test,T_test))

        # print("Accuracy from scratch BNN with momentum: ", sk.compute_accuracy(bnn2,X_test,T_test))

        # plt.title("BNN vs NN with eta={}".format(eta))
        BNN_errros = np.mean(np.array(grads).reshape(-1, 100), 1)
        # BNN_errros_2 = np.mean(np.array(BNN_errros_2).reshape(-1, 100), 1)
        # fig = plt.figure()
        plt.title('Learning rates')
        #
        # plt.title('Learning rate {}'.format(eta))
        plt.plot(BNN_errros,label='eta={}'.format(eta))

        # plt.plot(BNN_errros,label='normal')
        # plt.plot(BNN_errros_2,label='momentum')
        # plt.legend()
        # plt.show()
        # fig.savefig('/Users/vaevictis/Documents/As1/docs/images/momentum_plot_{}.png'.format(eta))
        #
        # plt.plot(BNN_errros_2,label='momentum')
    plt.legend()
    fig.savefig('/Users/vaevictis/Documents/As1/docs/images/NN_eta_vs_training_momentum.png')
    plt.show()








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