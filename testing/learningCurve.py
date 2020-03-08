from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np


#use full data no cross fold
def createLearningCurve(estimator, x, y, cv=None, name="", keras=False, is_estimator=True, scores=None):
    #if the estimator is the keras model
    if keras == True:
        #model already ran
        plt.plot(estimator.history['acc'])
        plt.plot(estimator.history['val_acc'])
        fname = "./results/" + name + '_learning_curve.png'
        plt.savefig(fname)
        return

    if is_estimator:
        train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))

    else:
        #read in scores[x[kfolds], y[kfolds], testx[kfolds], test_y[kfolds]]
        train_sizes = scores[0][0]
        test_scores = []
        train_scores = []
        
        '''
        for i in range(len(scores)):
            train_scores.append(scores[i][1])
            test_scores.append(scores[i][3])
        '''
        time_train = []
        time_test = []
        for i in range(len(scores)):
            #rolls over all train scores of ith run
            for j in range(len(scores[i][1])):
                #create time array 
                #add first time score
                if i == 0:
                    time_train.append([scores[i][1][j]])
                else:
                    train_scores[j].append(scores[i][1][j])
                
            
            #for 1 to batchNum test scores
            for j in range(len(scores[i][3])):
                #add first time score
                if i == 0:
                    time_test.append([scores[i][3][j]])
                else:
                    test_scores[j].append(scores[i][3][j])

            if i == 0:
                #two arrays of batchNum arrays each
                train_scores = time_train
                test_scores = time_test
        

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

    