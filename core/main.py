import utils
import skeleton as sk
from Perceptron import Perceptron
import numpy as np


# sk.run_part1()

# sk.run_part2()
#
train_X, train_T= sk.twospirals(250, noise=2, twist=800)
#
not_used_x, not_used_t, X_test, T_test  = utils.get_train_and_test_data(train_X,train_T,90)
#
# X,T = sk.twospirals(250, noise=0.6, twist=800)
#
sk.competition_load_weights_and_evaluate_X_and_T(X_test,T_test)
# for i in range(1):
#     sk.competition_train_from_scratch(X_test,T_test)
#
# # sk.gradient_check()