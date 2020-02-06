from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten
from tensorflow.keras import backend

import ml.crossFold as cfold

'''
    CNN has different input type 
    Each base in kmer is used separately as input 
    Then added features are similar to ones used in svm(i.e. mean/median)

'''


class createCNN():
    def __init__(self, x, y, nsplit):
        self.X = None
        self.Y = None
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
            for k in range(len(1, self.X[i])):
                row.append(self.X[i][k])

            self.X[i] = row

        print(self.X[0])
        #cross fold validation
        self.xtrain, self.ytrain, self.xtest, self.ytest = cfold.splitData(self.n, self.X, self.Y)
        
    
    def build_seq_model(self):
        model = Sequential()

        #add model layers
        model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(1,)))
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model


    def run_model(self):
        self.pre_process()
        model = self.build_seq_model()
        for i in range(len(self.xtrain)):
            model.fit(self.xtrain[i], self.ytrain[i], validation_data=(xtest, ytest), epochs=3)