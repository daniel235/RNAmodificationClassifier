from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pandas as pd
from statistics import mean, median
from keras.layers import LeakyReLU
import numpy as np
import platform
import os
import sys
import pseudoExtractor as ps

sys.path.insert(1, "./ml/")
sys.path.insert(3, "./testing/")
sys.path.insert(4, "/stats/")
import ml.svm as svm
import ml.knn as knn
import ml.cnn as cnn
import ml.logistic as logistic
import ml.fourier as fourier
import ml.rnn as rnn
import ml.sgd as sgd 
import signalExtractor as signal
import testing.learningCurve as lcurve
import stats.stats as stats


controlHela = pd.DataFrame()
pseudoHela = pd.DataFrame()
drp = [0, 2]
#continous file extraction
while(True):
    #start pseudoExtractor 
    choice1 = input("get control?(y/n)")
    if choice1 == 'y':
        temPcontrolHela = ps.get_control_Hela()
        controlHela = controlHela.append(temPcontrolHela.drop(drp, axis=1))
        
    choice2 = input("get pseudo?(y/n)")
    if choice2 == 'y':
        temPpseudoHela = ps.get_pseudo_Hela()
        pseudoHela = pseudoHela.append(temPpseudoHela.drop(drp, axis=1))
        

    if choice1 != 'y' and choice2 != 'y':
        break


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
prevIndexes = np.random.choice(len(controlHela), len(pseudoHela), replace=False)

#set length to 300(random choices)
totalControlKmerData = np.array(kmerData)
kmerData = np.array(kmerData)[prevIndexes]
print("size of ", len(kmerData))
total = len(pseudoHela) * 2
#indexes = np.random.choice(total, total, replace=False)

allKmerData = []

#randomize indexes
X = np.array(Xval)
Y = np.array(Yval)
print(len(X), len(Y))


#get all kmers
kmerList = []
for i in range(len(totalControlKmerData)):
    kmerList.append([totalControlKmerData[i][0]])

for i in range(len(pseudoKmerData)):
    kmerList.append([pseudoKmerData[i][0]])


le = preprocessing.LabelEncoder()

le.fit(kmerList)

print(le.classes_)

X = le.transform(kmerList)

X = X.reshape(-1, 1)

#onehot encode
enc = OneHotEncoder(handle_unknown='ignore')

enc.fit(X)



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


def getRnnData():
    index = np.random.choice(len(controlHela), len(pseudoHela), replace=False)
    kmerData = totalControlKmerData[index]
    rnnDatax = []
    rnnDatay = []
    kmers = []
    pkmers = []

    for i in range(len(pseudoKmerData)):
        kmers.append(kmerData[i][0])
        pkmers.append(pseudoKmerData[i][0])

    kmers = le.transform(kmers)
    kmers = kmers.reshape(-1, 1)
    pkmers = le.transform(pkmers)
    pkmers = pkmers.reshape(-1, 1)
    kmers_hot = enc.transform(kmers).toarray()
    pkmers_hot = enc.transform(pkmers).toarray()
    
    tempx = []
    for i in range(len(kmerData)):
        tempx = []
        #get mean median max min length
        tempx.append(median(kmerData[i][1:]))
        tempx.append(mean(kmerData[i][1:]))
        tempx.append(min(kmerData[i][1:]))
        tempx.append(max(kmerData[i][1:]))
        tempx.append(len(kmerData[i][1:]))
        '''
        #signal input
        for j in range(len(kmers_hot[i])):
            tempx.append(kmers_hot[i][j])

        tempx = np.array(tempx)
        #tempx.append(kmerData[i][1:])
        tempx = np.concatenate([tempx, kmerData[i][1:]])
        '''
        rnnDatax.append(tempx)
        rnnDatay.append([0])
        
    tempx = []
    for i in range(len(pseudoKmerData)):
        #signal input
        tempx = []
        tempx.append(median(pseudoKmerData[i][1:]))
        tempx.append(mean(pseudoKmerData[i][1:]))
        tempx.append(min(pseudoKmerData[i][1:]))
        tempx.append(max(pseudoKmerData[i][1:]))
        tempx.append(len(pseudoKmerData[i][1:]))
        '''
        for j in range(len(pkmers_hot[i])):
            tempx.append(pkmers_hot[i][j])
        
        tempx = np.array(tempx)
        
        #tempx.append(pseudoKmerData[i][1:])
        tempx = np.concatenate([tempx, pseudoKmerData[i][1:]])
        '''
        rnnDatax.append(tempx)
        rnnDatay.append([1])

    

    return rnnDatax, rnnDatay


def getSvmData():
    index = np.random.choice(len(controlHela), 1000, replace=False)
    kmerData = totalControlKmerData[index]
    svmDatax = []
    svmDatay = []

    for i in range(len(kmerData)):
        #signal input
        row = []
        row.append(len(kmerData[i]) - 1)
        for j in range(1, len(kmerData[i])):
            row.append(kmerData[i][j])

        svmDatax.append(row)
        svmDatay.append([0])


    for i in range(len(pseudoKmerData)):
        #signal input
        row = []
        row.append(len(pseudoKmerData[i]) - 1)
        for j in range(1, len(pseudoKmerData[i])):
            row.append(pseudoKmerData[i][j])

        svmDatax.append(row)
        svmDatay.append([1])

    return svmDatax, svmDatay

##################################################################


###############   Get Signal Data #############

#get padded signal data
#x, y = signal.signal_data(allKmerData, Yval)
#x, y = getCnnData()

#get density plot of signal data
#stats.get_signal_distribution(x, y)

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
#model.runSVM(x, y, 2)
#model.tuneParameters(x, y)
model.test_accuracy(x, y)
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
print("y", y)
kneighbors = knn.createKNN()
#kneighbors.runKNN(x, y, 3)

lcurve.createLearningCurve(kneighbors.knn, x, y, name="knn")

##################   Call CNN   #######################

x, y = getCnnData()
x, y = signal.signal_data(x, y)
model = cnn.createCNN(x, y, 10)
#model.run_model()
model = model.single_run(f=80, a='relu', k=20)

lcurve.createLearningCurve(model, x, y, keras=True, name="CNN")

#call learning curve

model.pre_process()
estimator = model.build_seq_model()
lcurve.createLearningCurve(estimator, model.X, model.Y)


#################  Call Logistic Regression  #############

x, y = getRnnData()
x, y = signal.signal_data(x, y)
indexes = np.random.choice(len(x), len(x), replace=False)
x = np.array(x)
y = np.array(y)
x = x[indexes]
y = y[indexes]
l = logistic.logRegression()
l.fit(x, y)
lcurve.createLearningCurve(l.reg, x, y, name="LogReg")


x, y = getCnnData()
paths = os.getcwd()
if platform.system() == 'Windows':
    paths = paths + "\data\images"
else:
    paths = paths + "/data/images"

#fourier.get_images(x, y)
cnn.createCNN(x, y, 3).ImageCNN(paths)
#x, y = signal.signal_data(x, y)



stats.std_deviation_distribution(x, y)
stats.signal_amplitude_mean(x, y)

'''
x, y = getRnnData()
x, y = signal.signal_data(x, y)
#fourier.read_signal(x, y)
#rnet = rnn.createRNN()
#rnet.createNet(x, y)
lstmNet = rnn.createRNN(x, y)
batch, batchy, test, test_out = lstmNet.createBatchData()
model = lstmNet.prepareRNN(batch, batchy)
'''
#lstmNet.runLSTM(model)

'''
