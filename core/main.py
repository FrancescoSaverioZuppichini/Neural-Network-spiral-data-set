from Perceptron import Perceptron
from NeuralNetwork import NeuralNetwork as NN
import skeleton as sk
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
from utils import timing

# X,T = sk.get_part1_data()

np.random.seed(1)
MAX_ITER = 15
STEP = 100

@timing
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

model = Perceptron()

# total_results, grads_average, errros_average, total_grads, total_erros = full_training(model,0.2,X,T, MAX_ITER, False, 0.5, 0)


net = NN()
X,T = sk.twospirals()

# for x in X:
#     net.forward(x)
grads, costs = net.train(X,T)

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