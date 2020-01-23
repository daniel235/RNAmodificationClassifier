from sklearn import svm

class createSVM():
    '''
        Class takes optional kernel
    '''
    def __init__(self, kernelType=None):
        self.clf = svm.SVC()
        self.X = None
        self.Y = None


    def runSVM(self, X, Y):
        self.clf.fit(X,Y)
    

    def tuneParameters(self):
        pass




    