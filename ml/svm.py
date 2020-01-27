from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
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
        for i in range(n_splits):
            self.clf.fit(self.trainX[i], self.trainY[i])
            #test accuracies
            predictions = self.clf.predict(self.testX)
            self.accuracies.append(accuracy_score(self.testY, predictions))
            print(self.accuracies[i])


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


