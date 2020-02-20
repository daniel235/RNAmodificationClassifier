from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pandas as pd
from statistics import mean, median
from keras.layers import LeakyReLU
import numpy as np
import sys
import pseudoExtractor as ps

sys.path.insert(1, "./ml/")
sys.path.insert(3, "./testing/")
sys.path.insert(4, "/stats/")
import ml.svm as svm
import ml.knn as knn
import ml.cnn as cnn
import ml.logistic as logistic
import signalExtractor as signal
import testing.learningCurve as lcurve
import stats.stats as stats


#start pseudoExtractor 
controlHela, pseudoHela = ps.get_Hela()

#omit file name
drp = [0, 2]
controlHela = controlHela.drop(drp, axis=1)
pseudoHela = pseudoHela.drop(drp, axis=1)


################### Extract data ###########################
kmerData = []

for i in range(len(controlHela)):
    kmer = controlHela.iloc[i, 0]
    kmerData.append([kmer])

    values = controlHela.iloc[i, 1]

    sig = ""
    for j in range(len(values)):
        if values[j] == '_':
            #convert to int
            kmerData[i].append(int(sig))
            sig = ""

        elif j == (len(values) - 1):
            sig += values[j]
            kmerData[i].append(int(sig))
            sig = ""

        else:
            sig += values[j]
        


pseudoKmerData = []
for i in range(len(pseudoHela)):
    kmer = pseudoHela.iloc[i, 0]
    pseudoKmerData.append([kmer])

    values = pseudoHela.iloc[i, 1]
    sig = ""
    for j in range(len(values)):
        if values[j] == '_':
            #convert to int
            pseudoKmerData[i].append(int(sig))
            sig = ""

        elif j == (len(values) - 1):
            sig += values[j]
            pseudoKmerData[i].append(int(sig))
            sig = ""

        else:
            sig += values[j]

#############################################################

X = []
Xval = []
Y = []
Yval = []

#get random indexes
prevIndexes = np.random.choice(len(controlHela), 364, replace=False)

#set length to 300(random choices)
totalControlKmerData = np.array(kmerData)
kmerData = np.array(kmerData)[prevIndexes]
print("size of ", len(kmerData))
total = 364 + len(pseudoHela)
#indexes = np.random.choice(total, total, replace=False)

allKmerData = []

#randomize indexes
X = np.array(Xval)
Y = np.array(Yval)
print(len(X), len(Y))


################# Data inputs for classifiers  ########################

def getKnnData():
    #get 2000 signal instances from controls
    index = np.random.choice(len(controlHela), len(pseudoHela), replace=False)
    kmerData = totalControlKmerData[index]
    knnDatax = []
    knnDatay = []
    for i in range(len(kmerData)):
        #signal input
        knnDatax.append(kmerData[i][1:])
        knnDatay.append([0])


    for i in range(len(pseudoKmerData)):
        #signal input
        knnDatax.append(pseudoKmerData[i][1:])
        knnDatay.append([1])

    return knnDatax, knnDatay


def getCnnData():
    index = np.random.choice(len(controlHela), len(pseudoHela), replace=False)
    kmerData = totalControlKmerData[index]
    CnnDatax = []
    CnnDatay = []
    for i in range(len(kmerData)):
        #signal input
        CnnDatax.append(kmerData[i][1:])
        CnnDatay.append([0])


    for i in range(len(pseudoKmerData)):
        #signal input
        CnnDatax.append(pseudoKmerData[i][1:])
        CnnDatay.append([1])

    return CnnDatax, CnnDatay


def getSvmData():
    index = np.random.choice(len(controlHela), len(pseudoHela), replace=False)
    kmerData = totalControlKmerData[index]
    svmDatax = []
    svmDatay = []

    for i in range(len(kmerData)):
        #signal input
        svmDatax.append(kmerData[i][1:])
        svmDatay.append([0])


    for i in range(len(pseudoKmerData)):
        #signal input
        svmDatax.append(pseudoKmerData[i][1:])
        svmDatay.append([1])

    return svmDatax, svmDatay

##################################################################


###############   Get Signal Data #############

#get padded signal data
#x, y = signal.signal_data(allKmerData, Yval)

#get density plot of signal data
#stats.get_signal_distribution(allKmerData, Y)

##################   Call SVM   #######################

'''
x, y = getSvmData()
x, y = signal.signal_data(x, y)
indexes = np.random.choice(len(x), len(x), replace=False)
x = np.array(x)
y = np.array(y)
x = x[indexes]
y = y[indexes]
model = svm.createSVM()
#estimator = model.pipelineSVM()
model.runSVM(x, y, 3)

#y = np.array(y)
#y = y.reshape((len(x), ))

#lcurve.createLearningCurve(estimator, np.array(x), y, name="svm_LinearSVC")
lcurve.createLearningCurve(model.clf, x, y, name="SVC")

##################   Call KNN   #######################

x, y = getKnnData()
x, y = signal.signal_data(x, y)
indexes = np.random.choice(len(x), len(x), replace=False)
x = np.array(x)
y = np.array(y)
x = x[indexes]
y = y[indexes]
kneighbors = knn.createKNN()
kneighbors.runKNN(x, y, 3)

#lcurve.createLearningCurve(kneighbors.knn, x, y, name="knn")

##################   Call CNN   #######################
'''

x, y = getCnnData()
x, y = signal.signal_data(x, y)
model = cnn.createCNN(x, y, 3)
#model.run_model()
model = model.single_run(f=80, a='tanh', k=20)
lcurve.createLearningCurve(model, x, y, keras=True, name="CNN")

#call learning curve
'''
model.pre_process()
estimator = model.build_seq_model()
lcurve.createLearningCurve(estimator, model.X, model.Y)


#################  Call Logistic Regression  #############

x, y = getSvmData()
x, y = signal.signal_data(x, y)
l = logistic.logRegression()
l.fit(x, y)


x, y = getSvmData()
#x, y = signal.signal_data(x, y)
stats.signal_amplitude_mean(x, y)
'''