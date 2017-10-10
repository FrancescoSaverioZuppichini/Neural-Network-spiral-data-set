import numpy as np

def BacktrackingLineSearch(f, df, x, p, df_x=None, f_x=None, args=(),
                           alpha=0.0001, beta=0.9, eps=_epsilon, Verbose=False):
    """
    Backtracking linesearch
    f: function
    x: current point
    p: direction of search
    df_x: gradient at x
    f_x = f(x) (Optional)
    args: optional arguments to f (optional)
    alpha, beta: backtracking parameters
    eps: (Optional) quit if norm of step produced is less than this
    Verbose: (Optional) Print lots of info about progress

    Reference: Nocedal and Wright 2/e (2006), p. 37

    Usage notes:
    -----------
    Recommended for Newton methods; less appropriate for quasi-Newton or conjugate gradients
    """

    if f_x is None:
        f_x = f(x, *args)
    if df_x is None:
        df_x = df(x, *args)

    derphi = np.dot(df_x, p)


    stp = 1.0
    fc = 0
    len_p = np.linalg.norm(p)

    # Loop until Armijo condition is satisfied
    while f(x + stp * p, *args) > f_x + alpha * stp * derphi:
        stp *= beta
        fc += 1
        if Verbose:
            print('linesearch iteration', fc, ':', stp, f(x + stp * p, *args), f_x + alpha * stp * derphi)
        if stp * len_p < eps:
            print('Step is  too small, stop')
            break
    # if Verbose: print 'linesearch iteration 0 :', stp, f_x, f_x

    if Verbose:
        print('linesearch done')
    # print fc, 'iterations in linesearch'
    return stp