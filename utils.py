import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print(' function took {:0.3f} ms'.format(float(time2-time1)*1000.0))
        return ret
    return wrap