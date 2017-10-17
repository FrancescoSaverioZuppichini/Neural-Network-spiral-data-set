from Perceptron import Perceptron
import utils
import skeleton as sk

import time
import numpy as np
from threading import Thread


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


# sk.run_part1()
# sk.run_part2()


X_train,T_train= sk.twospirals(400, noise=0.7, twist=810)
# sk.plot_data(X_train,T_train)
Dio, Cane, X_test, T_test  = utils.get_train_and_test_data(X_train,T_train,90)

X,T = sk.twospirals(250, noise=0.6, twist=800)


sk.competition_load_weights_and_evaluate_X_and_T(X_test,T_test)
# for i in range(1):
#     sk.competition_train_from_scratch(X_test,T_test)
