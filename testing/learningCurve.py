from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np


#use full data no cross fold
def createLearningCurve(estimator, x, y):
    train_sizes, train_scores, test_scores, fit_times = learning_curve(estimator, x, y, train_sizes=np.linspace(.1, 1.0, 5))

    #create plots
    _, plots = plt.subplots(1, 3, figsize(20, 5))


    #plot curves
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    plots[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plots[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

    #plot train
    plots[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    
    #plot test
    plots[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross validation score")

    plots[0].legend(loc="best")

    