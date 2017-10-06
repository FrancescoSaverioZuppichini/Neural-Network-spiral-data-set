from Perceptron import Perceptron
import skeleton as sk
import matplotlib.pyplot as plt
import numpy as np
from utils import timing

X,T = sk.get_part1_data()


MAX_ITER = 200
STEP = 100

@timing
def full_training(model, learning_rate, inputs, targets, maxIter, momentum, beta):
    z = 1
    results = []
    grads = []
    errors = []
    total_grads = []
    ## Implement
    for n in range(maxIter):
        results.append([])
        errors.append([])

        y, grad, grads = sk.train_one_step(model, learning_rate, inputs, targets, momentum, beta)
        total_grads.append(grads)

        # append average gradient
        grads.append(grad)
        # append last prediction for this train
        results.append(y)

    return results,grads

results, grads = full_training(Perceptron(),0.5,X,T, MAX_ITER, False, 0.5)

plt.plot(T, 'ro',label="T")

lastResult = results[len(results) - 1]



# results = [results[i] for i in range(len(results)) if i % STEP == 0]
# errors = [errors[i] for i in range(len(errors)) if i % STEP == 0]


plt.plot(lastResult, label='best')


plt.legend()
plt.show()

plt.plot(grads, label='grad')

plt.legend()
plt.show()
