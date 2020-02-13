from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pandas as pd
from statistics import mean, median
import numpy as np
import sys
import pseudoExtractor as ps

sys.path.insert(1, "./ml/")
sys.path.insert(2, "./scripts/")
sys.path.insert(3, "./testing/")
sys.path.insert(4, "/stats/")
import ml.svm as svm
import ml.knn as knn
import ml.cnn as cnn
import signalExtractor as signal
import scripts.complete as complete
import testing.learningCurve as lcurve
import stats.stats as stats


#start pseudoExtractor 
controlHela, pseudoHela = ps.get_Hela()

#omit file name
drp = [0, 2]
controlHela = controlHela.drop(drp, axis=1)
pseudoHela = pseudoHela.drop(drp, axis=1)

print(controlHela.iloc[0,1])


kmerData = []
#!cut data to 500
#for i in range(len(controlHela)):
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

X = []
Xval = []
Y = []
Yval = []

#get random indexes
prevIndexes = np.random.choice(len(controlHela), 364, replace=False)

#set length to 300(random choices)
kmerData = np.array(kmerData)[prevIndexes]
print("size of ", len(kmerData))
total = 364 + len(pseudoHela)
#indexes = np.random.choice(total, total, replace=False)


#adding kmer 
for i in range(len(kmerData)):
    X.append(kmerData[i][0])


for i in range(len(pseudoKmerData)):
    X.append(pseudoKmerData[i][0])


allKmerData = []

for i in range(len(kmerData)):
    #signal cnn input
    allKmerData.append(kmerData[i][1:])
    #svm input
    Xval.append([mean(kmerData[i][1:]), median(kmerData[i][1:]), max(kmerData[i][1:]), min(kmerData[i][1:]), np.std(kmerData[i][1:])])
    #cnn input
    #Xval.append([kmerData[i][0], mean(kmerData[i][1:]), median(kmerData[i][1:]), max(kmerData[i][1:]), min(kmerData[i][1:]), np.std(kmerData[i][1:])])
    Yval.append([0])


for i in range(len(pseudoKmerData)):
    #signal cnn input
    allKmerData.append(pseudoKmerData[i][1:])
    Xval.append([mean(pseudoKmerData[i][1:]), median(pseudoKmerData[i][1:]), max(pseudoKmerData[i][1:]), min(pseudoKmerData[i][1:]), np.std(pseudoKmerData[i][1:])])
    #Xval.append([kmerData[i][0], mean(pseudoKmerData[i][1:]), median(pseudoKmerData[i][1:]), max(pseudoKmerData[i][1:]), min(pseudoKmerData[i][1:]), np.std(pseudoKmerData[i][1:])])
    Yval.append([1])



#randomize indexes
X = np.array(Xval)
Y = np.array(Yval)
print(len(X), len(Y))

###############   Get Signal Data #############

#get padded signal data
x, y = signal.signal_data(allKmerData, Yval)

#get density plot of signal data
#stats.get_signal_distribution(allKmerData, Y)

##################   Call SVM   #######################
'''
model = svm.createSVM()
estimator = model.pipelineSVM()
#model.runSVM(X, Y, 3)


Y = np.array(Y)
Y = Y.reshape((len(X), ))
lcurve.createLearningCurve(estimator, np.array(X), Y, name="svm_LinearSVC")
lcurve.createLearningCurve(model.clf, np.array(X), Y, name="SVC")
'''
##################   Call KNN   #######################
'''
kneighbors = knn.createKNN()
#kneighbors.runKNN(3, X, Y)

lcurve.createLearningCurve(kneighbors.knn, X, Y, name="knn")
'''
##################   Call CNN   #######################

model = cnn.createCNN(x, y, 3)
model.run_model()


#call learning curve
'''
model.pre_process()
estimator = model.build_seq_model()
lcurve.createLearningCurve(estimator, model.X, model.Y)
'''
