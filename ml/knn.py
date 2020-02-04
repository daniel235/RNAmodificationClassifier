from sklearn.neighbors import KNeighborsClassifier
import ml.crossFold
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


    def runKNN(self, nsplit, x, y):
        self.X = x
        self.Y = y
        #cross fold
        self.trainX, self.trainY, self.testX, self.testY = ml.crossFold.splitData(nsplit, x, y)
        for i in range(nsplit):    
            self.knn.fit(self.trainX[i], self.trainY[i])
            self.accuracy.append(self.knn.score(self.testX[i], self.testY[i]))
            print("accuracy ", self.accuracy[i])

        #save scores
        self.saveAccuracy(self.accuracy)


    def saveAccuracy(self, acc):
        with(open("./results/knnAccuracy.txt", 'a+')) as f:
            for i in range(len(acc)):
                lines = str(date.today()) + " " + str(acc[i]) + "\n"
                f.write(lines)
        
    