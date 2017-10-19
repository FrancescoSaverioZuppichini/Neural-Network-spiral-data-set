import utils
import skeleton as sk


# sk.run_part1()
# sk.run_part2()


train_X, train_T= sk.twospirals(250, noise=0.6, twist=300)
#
not_used_x, not_used_t, X_test, T_test  = utils.get_train_and_test_data(train_X,train_T,90)

sk.competition_load_weights_and_evaluate_X_and_T(X_test,T_test)
# sk.competition_train_from_scratch(X_test,T_test)
