from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
import pickle
import matplotlib.pyplot as plt
import numpy as np
import platform


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
        self.featureImportance()
        self.splitData(n_splits)
        with open("./results/accuracies.txt", 'w+') as f:
            for i in range(n_splits):
                self.clf.fit(self.trainX[i], self.trainY[i])
                #test accuracies
                predictions = self.clf.predict(self.testX[i])
                for l in range(len(predictions)):
                    print("pred ", predictions[l])

                for a in range(len(self.testY[i])):
                    print("actual ", self.testY[i][a])

                self.accuracies.append(accuracy_score(self.testY[i], predictions))
                print(self.accuracies[i])
                #save accuracies
                f.write(str(self.accuracies[i]))
                f.write("\n")

        
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


    #find out what features are valuable
    def featureImportance(self):
        #tree = DecisionTreeClassifier(criterion='entropy')
        model = ExtraTreesClassifier()
        model.fit(self.X, self.Y)
        print(model.feature_importances_)
    

    #plot feature importance
    def f_importances(coef, names):
        imp = coef
        imp,names = zip(*sorted(zip(imp,names)))
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)
        if platform.system() == 'Windows':
            plt.show()

