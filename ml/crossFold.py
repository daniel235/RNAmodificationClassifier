from sklearn.model_selection import KFold


def splitData(splits, X, Y):
        #returned data structs
        trainX = []
        trainY = []
        testX = []
        testY = []
        kf = KFold(n_splits=splits, shuffle=True)
        kf.get_n_splits(X)
        for train_index, test_index in kf.split(X):
            #print indexes
            trainX.append(X[train_index])
            testX.append(X[test_index])
            trainY.append(Y[train_index])
            testY.append(Y[test_index])

        return trainX, trainY, testX, testY