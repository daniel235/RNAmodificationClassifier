from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold, GridSearchCV

from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
import numpy as np
import platform


class createSVM():
    '''
        Class takes optional kernel
    '''
    def __init__(self, kernelType=None):
        self.clf = svm.SVC(kernel='poly', C=3.0, degree=4)
        self.polysvm = None
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
        #reshape y
        self.Y = self.Y.reshape((len(self.X),))
        #self.featureImportance()
        self.splitData(n_splits)
        with open("./results/accuracies.txt", 'w+') as f:
            for i in range(n_splits):
                self.clf.fit(self.trainX[i], self.trainY[i])
                #test accuracies
                predictions = self.clf.predict(self.testX[i])
                print(confusion_matrix(self.testY[i], predictions))
                
                #write accuracies 
                self.accuracies.append(accuracy_score(self.testY[i], predictions))
                print(self.accuracies[i])
                #save accuracies
                f.write(str(self.accuracies[i]))
                f.write("\n")
                
                #write poly svm accuracies
                '''
                self.pipelineSVM(self.trainX[i], self.trainY[i], Cval=10, max=10000000)
                self.polysvm.fit(self.trainX[i], self.trainY[i])
                predictions = self.polysvm.predict(self.testX[i])
                print(confusion_matrix(self.testY[i], predictions))
                print(self.polysvm.score(self.testX[i], self.testY[i]))
                #write accuracies
                polyLine = "pipeline accuracy: " + str(accuracy_score(self.testY[i], predictions)) + "\n"
                f.write(polyLine)
                '''

        #save model
        fname = "./results/svm"
        with open(fname, 'wb+') as f:
            pickle.dump(self.clf, f)


    def tuneParameters(self, x, y):
        #parameters
        tuned_parameters = [{'kernel': ['poly'], 'C': [1, 3, 5, 200], 'gamma': [1e-3, 1e-4], 'degree': [3, 4, 5, 6]}]
        #use grid search
        clf = GridSearchCV(svm.SVC(), tuned_parameters)
        clf.fit(x, y)
        #write best params to file
        with open("./results/svm_best_results.txt", 'a+') as f:
            line = str(clf.best_params_)
            f.write(line)


    def test_accuracy(self, x, y):
        #load svm
        pfile = open("./results/svm", 'rb')
        svmModel = pickle.load(pfile)
        predictions = svmModel.predict(x)
        print(accuracy_score(y, predictions))




    def splitData(self, splits):
        kf = KFold(n_splits=splits, shuffle=True)
        kf.get_n_splits(self.X)
        for train_index, test_index in kf.split(self.X):
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

    
    def pipelineSVM(self, polydegree=3, Cval=10, linearloss="hinge", max=1000):
        self.polysvm = Pipeline([
            ("poly features", PolynomialFeatures(degree=polydegree)),
            ("scaler", StandardScaler()),
            ("svm_clf", svm.LinearSVC(C=Cval, loss=linearloss, verbose=1, max_iter=max))
        ])

        return self.polysvm
        #self.polysvm.fit(x, y)

    
    def plotSupportVectors(self, svm):
        decisionFunction = svm.decision_function(X)
        

    def scaleData(self, x, y):
        pass