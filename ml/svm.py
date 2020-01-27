from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pickle
import numpy as np


class createSVM():
    '''
        Class takes optional kernel
    '''
    def __init__(self, kernelType=None):
        self.clf = svm.SVC()
        self.X = None
        self.Y = None
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []
        self.accuracies = []


    def runSVM(self, X, Y, n_splits):
        self.X = np.array(X)
        self.Y = np.array(Y)
        
        self.splitData(n_splits)
        with open("./results/accuracies.txt", 'w+') as f:
            for i in range(n_splits):
                self.clf.fit(self.trainX[i], self.trainY[i])
                #test accuracies
                predictions = self.clf.predict(self.testX[i])
                self.accuracies.append(accuracy_score(self.testY[i], predictions))
                print(self.accuracies[i])
                #save accuracies
                f.write(self.accuracies[i])

        #save model
        fname = "./results/svm"
        with open(fname, 'wb+') as f:
            pickle.dump(self.clf, f)


    def tuneParameters(self):
        pass


    def splitData(self, splits):
        kf = KFold(n_splits=splits, shuffle=True)
        kf.get_n_splits(self.X)
        for train_index, test_index in kf.split(self.X):
            #print indexes
            print(train_index)
            print(test_index)
            self.trainX.append(self.X[train_index])
            self.testX.append(self.X[test_index])
            self.trainY.append(self.Y[train_index])
            self.testY.append(self.Y[test_index])


