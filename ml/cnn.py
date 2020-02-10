from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten, Dropout, MaxPooling1D
from keras.utils.np_utils import to_categorical
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
            '''
            for j in range(len(self.X[i][0])):
                base = self.X[i][0][j] 
                if base == 'A':
                    row.append(0)
                    row.append(0)
                    row.append(0)
                    row.append(1)

                elif base == 'C':
                    row.append(0)
                    row.append(0)
                    row.append(1)
                    row.append(0)

                elif base == 'G':
                    row.append(0)
                    row.append(1)
                    row.append(0)
                    row.append(0)

                else:
                    row.append(1)
                    row.append(0)
                    row.append(0)
                    row.append(0)
            '''
            #restructure X[i] row
            
            for k in range(1, len(self.X[i])):
                print("data ", self.X[i][k])
                row.append(self.X[i][k])
            
            row = np.array(row)
            print("shape of new row ", row.shape)
            newX.append(row)
            row = []

        self.X = np.array(newX)
        sameY = self.Y
        #create data dimensions for cnn
        self.X = np.expand_dims(self.X, axis=2)
        self.Y = to_categorical(self.Y)

        #cross fold validation
        self.xtrain, self.ytrain, self.xtest, self.ytest = cfold.splitData(self.n, newX, sameY)
        print("shape ", np.array(self.xtrain[0]).shape)
        
        for i in range(len(self.xtrain)):
            self.xtrain[i] = np.expand_dims(self.xtrain[i], axis=2)
            self.xtest[i] = np.array(self.xtest[i])
            self.xtest[i] = np.expand_dims(self.xtest[i], axis=2)
            #convert to one hot
            self.ytrain[i] = to_categorical(self.ytrain[i])
            self.ytest[i] = to_categorical(self.ytest[i])
        

    def build_seq_model(self):
        model = Sequential()
        model.score = "accuracy"
        n_samples, n_feats = self.xtrain[0].shape[1], self.xtrain[0].shape[2]
        print("shape ", n_samples, n_feats)
        #[[],[],[]]
        #add model layers
        model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(n_samples, n_feats)))
        model.add(Conv1D(32, kernel_size=2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model


    def run_model(self):
        self.pre_process()
        model = self.build_seq_model()
        
        for i in range(len(self.xtrain)):
            print("i ", i)
            model.fit(self.xtrain[i], self.ytrain[i], epochs=10)
            print("after model fit")
            #validation_data=(self.xtest[i], self.ytest[i])
            _, accuracy = model.evaluate(self.xtest[i], self.ytest[i])
            print("acc ", accuracy)
        