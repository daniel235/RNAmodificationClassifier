from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np


#use full data no cross fold
def createLearningCurve(estimator, x, y, cv=None, name="", keras=False):
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))

    #create plots
    _, plots = plt.subplots(figsize=(20, 5))

    #set axis names 
    plots.set_xlabel("Training examples")
    plots.set_ylabel("Score")

    #plot curves
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    #fit_times_mean = np.mean(fit_times, axis=1)
    #fit_times_std = np.std(fit_times, axis=1)

    plots.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plots.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

    #plot train
    plots.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    #plot test
    plots.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross validation score")
    fname = "./results/" + name + '_learning_curve.png'
    plt.savefig(fname)

    plots.legend(loc="best")

    #if the estimator is the keras model
    if keras:
        #model already ran
        plt.plot(estimator.history['acc'])
        plt.plot(estimator.history['val_acc'])
        fname = "./results/" + name + '_learning_curve.png'
        plt.savefig(fname)