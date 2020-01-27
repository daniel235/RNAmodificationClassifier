from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pandas as pd
from statistics import mean
import numpy as np
import sys
import pseudoExtractor as ps

sys.path.insert(1, "./ml/")
import ml.svm as svm

#start pseudoExtractor 
controlHela, pseudoHela = ps.get_Hela()

#omit file name
drp = [0, 2]
controlHela = controlHela.drop(drp, axis=1)
pseudoHela = pseudoHela.drop(drp, axis=1)

print(controlHela.iloc[0,1])

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

X = []
Xval = []
Y = []
Yval = []
'''
for i in range(len(kmerData)):
    X.append(kmerData[i][0])
    Xval.append(kmerData[i][1:])
    Y.append(0)


for i in range(len(pseudoKmerData)):
    X.append(pseudoKmerData[i][0])
    Xval.append(pseudoKmerData[i][1:])
    Y.append(1)
'''
for i in range(len(kmerData)):
    for j in range(1, len(kmerData[i])):
        Xval.append([kmerData[i][j]])
        Yval.append([0])

for i in range(len(pseudoKmerData)):
    for j in range(1, len(pseudoKmerData[i])):
        Xval.append([pseudoKmerData[i][j]])
        Yval.append([1])

'''
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

Xinput = []
#insert signal Feature
for i in range(len(X)):
    Xinput.append([list(X[i]), mean(Xval[i])])

X = Xinput
'''
X = Xval
Y = Yval
print(len(X), len(Y))


##################   Call SVM   #######################
model = svm.createSVM()
model.runSVM(X, Y)

