from sklearn.model_selection import learning_curve

def createLearningCurve(svm, x, y):
    lc = learning_curve(svm, x, y)