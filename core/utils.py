import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        max_iter = args[4]
        ret = f(*args)
        time2 = time.time()
        total_time = float(time2-time1)*1000.0
        print('--------------------------')
        print('Finish after {:0.3f} ms'.format(total_time,max_iter))
        print('{} iterations'.format(max_iter))
        print('Learning rage {}'.format(args[3]))
        print('{:0.3f} iterations per seconds'.format(max_iter/total_time))
        return ret
    return wrap


def get_data(*args, data_func, train_ratio=100):

    X, T = data_func(*args)

    size = int((len(X) / 100) * train_ratio)

    X_train, T_train = X[:size], T[:size]
    X_test, T_test = X[size:], T[size:]

    return X_train, T_train, X_test, T_test