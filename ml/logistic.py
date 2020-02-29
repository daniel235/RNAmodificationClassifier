from sklearn.linear_model import LogisticRegression
import ml.crossFold as cfold
import numpy as np

class logRegression():
    def __init__(self):
        self.reg = LogisticRegression(random_state=42, max_iter=1000000)
        

    def fit(self, x, y):
        self.xtrain, self.ytrain, self.xtest, self.ytest = cfold.splitData(3, x, y)

        for i in range(len(self.xtrain)):
            self.reg.fit(self.xtrain[i], self.ytrain[i])
            #t, ty = cfold.getEvenTestData(self.xtest[i], self.ytest[i])
            t, ty = self.xtest[i], self.ytest[i]
            predictions = self.reg.predict(t)
            accuracy = 0
            for i in range(len(ty)):
                if ty[i][0] == predictions[i]:
                    accuracy += 1

            accuracy = accuracy / len(ty)
            print(accuracy)
            accuracy = 0

    
