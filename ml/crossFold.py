from sklearn.model_selection import KFold
import numpy as np


def splitData(splits, X, Y):
        #returned data structs
        X = np.array(X)
        if type(Y).__module__ != 'numpy':
            Y = np.array(Y)

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


def getEvenTestData(testx, test_output, limit=0):
    final_test_data = []
    final_test_y = []

    #get count of pseudo in test data
    for i in range(len(testx)):
        if test_output[i][0] == 1:
            limit += 1

    print("limit ", limit)


    test_size_control = limit
    test_size_pseudo = limit

    for i in range(len(testx)):
        if test_size_control > 0:
            if test_output[i][0] == 0:
                test_size_control -= 1
                final_test_data.append(testx[i])
                final_test_y.append(test_output[i])


        if test_size_pseudo > 0:
            if test_output[i][0] == 1:
                test_size_pseudo -= 1
                final_test_data.append(testx[i])
                final_test_y.append(test_output[i])

        if test_size_control == 0 and test_size_pseudo == 0:
            return final_test_data, final_test_y

    return final_test_data, final_test_y
        