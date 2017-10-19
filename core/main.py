import utils
import skeleton as sk


# sk.run_part1()
# sk.run_part2()

train_X, train_T= sk.twospirals(250, noise=0.6, twist=800)

res  = utils.get_train_and_test_data(train_X,train_T,80)

sk.competition_load_weights_and_evaluate_X_and_T(res[2],res[3])
# sk.competition_train_from_scratch(X_test,T_test)
