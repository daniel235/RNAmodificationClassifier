from sklearn import svm
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

    def runSVM(self, X, Y):
        self.X = X
        self.Y = Y
        #flatten data
        X = np.array(X)
        Y = np.array(Y)
        #X = X.reshape(1, -1)
        print(X.shape)
        print(X)
        #nsamples, nx, ny = X.shape
        #X2 = X.reshape(nsamples, nx*ny)
        self.clf.fit(X,Y)
        

    def tuneParameters(self):
        pass


    def splitData(self):
        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits(self.X)
        for train_index, test_index in kf.split(self.X):
            #print indexes
            print(train_index)
            print(test_index)
            self.trainX, self.testX = self.X[train_index], self.X[test_index]
            self.trainY, self.testY = self.Y[train_index], self.Y[test_index]


