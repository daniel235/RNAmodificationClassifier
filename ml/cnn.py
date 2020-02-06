from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten
import numpy as np

import ml.crossFold as cfold

'''
    CNN has different input type 
    Each base in kmer is used separately as input 
    Then added features are similar to ones used in svm(i.e. mean/median)

'''


class createCNN():
    def __init__(self, x, y, nsplit):
        self.X = x
        self.Y = y
        self.xtrain = []
        self.ytrain = []
        self.xtest = []
        self.ytest = []
        self.n = nsplit

    
    def pre_process(self):
        #separate out kmer
        #one hot 
        base = ""
        row = []
        newX = []
        #bases (A C G T[U])
        for i in range(len(self.X)):
            for j in range(len(self.X[i][0])):
                base = self.X[i][0][j] 
                if base == 'A':
                    row.append([0, 0, 0, 1])
                elif base == 'C':
                    row.append([0, 0, 1, 0])
                elif base == 'G':
                    row.append([0, 1, 0, 0])
                else:
                    row.append([1, 0, 0, 0])

            #restructure X[i] row
            for k in range(1, len(self.X[i])):
                row.append(self.X[i][k])

            newX.append(row)
            row = []

        self.X = newX

        #cross fold validation
        self.xtrain, self.ytrain, self.xtest, self.ytest = cfold.splitData(self.n, self.X, self.Y)
        print("shape ", np.array(self.xtrain[0]).shape)
    
    def build_seq_model(self):
        model = Sequential()

        #add model layers
        model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(9, 1)))
        model.add(Conv1D(32, kernel_size=2, activation='relu'))
        print(model.summary())
        model.add(Flatten())
        model.add(Dense(1, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model


    def run_model(self):
        self.pre_process()
        model = self.build_seq_model()
        
        for i in range(len(self.xtrain)):
            self.xtrain[i] = np.expand_dims(self.xtrain[i], axis=2)
            self.xtest[i] = np.array(self.xtest[i])
            self.xtest[i] = np.expand_dims(self.xtest[i], axis=2)
            model.fit(self.xtrain[i], self.ytrain[i], validation_data=(self.xtest[i], self.ytest[i]), epochs=3)
        