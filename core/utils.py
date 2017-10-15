import time
from random import choice
import numpy as np

def timing(f):
    def wrap(*args):
        time1 = time.time()
        max_iter = args[3]
        ret = f(*args)
        time2 = time.time()
        total_time = float(time2-time1)*1000.0
        print('--------------------------')
        print('Finish after {:0.3f} ms'.format(total_time,max_iter))
        print('{} iterations'.format(max_iter))
        print('Learning rage {}'.format(args[4]['eta']))
        if(len(args) > 5):
            print('Type {}'.format(args[5]))
        print('{:0.3f} iterations per seconds'.format(max_iter/total_time))
        return ret
    return wrap

def get_train_and_test_data(X,T,train_ratio=80):

    if (train_ratio > 100 or train_ratio < 0):
        raise Exception("train ratio must be >= 0 and <= 100")
    # convert to list so it is easier lol
    X = X.tolist()
    T = T.tolist()

    train_size = len(X)

    size = int((train_size / 100) * train_ratio)

    test_size = train_size - size

    X_test = []
    T_test = []

    indices = [x for x in range(train_size)]

    i = 0
    while(i < test_size):
        random_index = choice(indices)
        indices.remove(random_index)

        temp = X[random_index]

        X_test.append(temp)
        X[random_index] = None

        temp = T[random_index]

        T_test.append(temp)
        T[random_index] = None

        i += 1

    X = [x for x in X if x != None]
    T = [t for t in T if t != None]

    return np.array(X),np.array(T),np.array(X_test),np.array(T_test)

