from sklearn.neighbors import KNeighborsClassifier
import ml.crossFold
import numpy as np
from datetime import date

class createKNN():
    def __init__(self):
        self.knn = KNeighborsClassifier()
        self.X = None
        self.Y = None
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []
        self.accuracy = []


    def runKNN(self, x, y, nsplit):
        self.X = x
        self.Y = y
        #cross fold
        self.trainX, self.trainY, self.testX, self.testY = ml.crossFold.splitData(nsplit, x, y)
        for i in range(nsplit):   
            self.trainY[i] = np.array(self.trainY[i]).reshape((len(self.trainX[i], )))
            print(self.trainY[i].shape)
            self.knn.fit(self.trainX[i], self.trainY[i])
            #todo even test size
            testx, testoutput = ml.crossFold.getEvenTestData(self.testX[i], self.testY[i])
            print("len of test ", len(testx), " ", len(testoutput))
            self.accuracy.append(self.knn.score(testx, testoutput))
            print("accuracy ", self.accuracy[i])

        #save scores
        self.saveAccuracy(self.accuracy)


    def saveAccuracy(self, acc):
        with(open("./results/knnAccuracy.txt", 'a+')) as f:
            for i in range(len(acc)):
                lines = str(date.today()) + " " + str(acc[i]) + "\n"
                f.write(lines)
        
    