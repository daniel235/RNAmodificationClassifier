from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten, Dropout, MaxPooling1D, LeakyReLU
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam, Nadam
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
        self.xsignal = None
        self.ysignal = None

    
    def pre_process(self):
        #self.ysignal = to_categorical(self.ysignal)

        #cross fold validation
        self.xtrain, self.ytrain, self.xtest, self.ytest = cfold.splitData(self.n, self.xsignal, self.ysignal)
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
        n_samples, n_feats = self.xtrain[0].shape[1], self.xtrain[0].shape[2]
        print("shape ", n_samples, n_feats)
        #[[],[],[]]
        #add model layers
        model.add(Conv1D(64, kernel_size=2, input_shape=(n_samples, n_feats)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Conv1D(32, kernel_size=2))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=10))
        
        model.add(Flatten())
        model.add(Dense(50))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(2, activation='softmax'))

        #optimzer 
        nesterov = Nadam(lr=0.001)
        model.compile(optimizer=nesterov, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model


    def run_model(self):
        self.pre_process()
        model = self.build_seq_model()
        
        for i in range(len(self.xtrain)):
            model.fit(self.xtrain[i], self.ytrain[i], epochs=5)
            #validation_data=(self.xtest[i], self.ytest[i])
            _, accuracy = model.evaluate(self.xtest[i], self.ytest[i])
            print("acc ", accuracy)
        
    
    def signal_data(self, x, y, timestep=120):
        xinputs = []
        youtputs = []
        signal = []
        currentCount = 0
        currentY = 0
        i = 0
        while(i < len(x)):
            #input raw signal data until signal array full
            for j in range(timestep):
                #if you run out of signal move to the next x line
                if currentCount >= len(x[i]):
                    i += 1
                    currentCount = 0
                    #if you run out of control samples 
                    if(i < len(y)):
                        #if next row is another y reset signal 
                        if y[i] != currentY:
                            currentY = y[i]
                            #reset signal array 
                            signal = []
                            continue
                    
                    else:
                        break

                signal.append(x[i][currentCount])
                currentCount += 1

            #after 30 signals add to xinput
            if len(signal) == timestep:
                xinputs.append(np.array(signal))
                youtputs.append(currentY)

            signal = []

        #remove last row
        xinputs.pop()
        #remove first row
        xinputs.pop(0)
        print(xinputs)
        self.xsignal = np.array(xinputs)
        #reshape
        self.xsignal.reshape((self.xsignal.shape[0], timestep))
        #remove first item
        print("x ", self.xsignal[1])
        print("xshape ", self.xsignal.shape)
        self.ysignal = np.array(youtputs)
