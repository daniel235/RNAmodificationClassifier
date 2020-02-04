from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pandas as pd
from statistics import mean, median
import numpy as np
import sys
import pseudoExtractor as ps

sys.path.insert(1, "./ml/")
sys.path.insert(2, "./scripts/")
import ml.svm as svm
import ml.knn as knn
import scripts.complete as complete


#start pseudoExtractor 
controlHela, pseudoHela = ps.get_Hela()

#omit file name
drp = [0, 2]
controlHela = controlHela.drop(drp, axis=1)
pseudoHela = pseudoHela.drop(drp, axis=1)

print(controlHela.iloc[0,1])
complete.send_email("Test email")


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
prevIndexes = np.random.choice(len(controlHela), 460, replace=False)

#set length to 300(random choices)
kmerData = np.array(kmerData)[prevIndexes]
print("size of ", len(kmerData))
total = 360 + len(pseudoHela)
indexes = np.random.choice(total, total, replace=False)


for i in range(len(kmerData)):
    X.append(kmerData[i][0])


for i in range(len(pseudoKmerData)):
    X.append(pseudoKmerData[i][0])


le = preprocessing.LabelEncoder()
le.fit(X)
print(le.classes_)
X = le.transform(X)
X = X.reshape(-1, 1)

#onehot encode
enc = OneHotEncoder(handle_unknown='ignore', n_values=350)
enc.fit(X)
onehots = enc.transform(X).toarray()
X = onehots


for i in range(len(kmerData)):
    Xval.append([mean(kmerData[i][1:]), median(kmerData[i][1:]), max(kmerData[i][1:]), min(kmerData[i][1:])])
    Yval.append([0])


for i in range(len(pseudoKmerData)):
    Xval.append([mean(pseudoKmerData[i][1:]), median(pseudoKmerData[i][1:]), max(pseudoKmerData[i][1:]), min(pseudoKmerData[i][1:])])
    Yval.append([1])

'''
#insert one hot Feature
for i in range(len(Xval)):
    for j in range(len(X[i])):
        Xval[i].append(X[i][j])
'''

#randomize indexes
X = np.array(Xval)[indexes]
Y = np.array(Yval)[indexes]
print(len(X), len(Y))


##################   Call SVM   #######################

model = svm.createSVM()
model.runSVM(X, Y, 3)

##################   Call KNN   #######################
'''
kneighbors = knn.createKNN()
kneighbors.runKNN(3, X, Y)
'''