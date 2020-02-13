from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten, Dropout, MaxPooling1D, LeakyReLU, GlobalAveragePooling1D, ELU
from keras.activations import elu, selu, tanh, sigmoid, hard_sigmoid, linear
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam, Nadam, SGD, RMSprop, Adagrad, Adadelta, Adamax 
import numpy as np
import sys


import crossFold as cfold

'''
    CNN has different input type 
    Each base in kmer is used separately as input 
    Then added features are similar to ones used in svm(i.e. mean/median)

'''

#todo pad data
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
        self.xtrain, self.ytrain, self.xtest, self.ytest = cfold.splitData(self.n, self.X, self.Y)
        
        for i in range(len(self.xtrain)):
            self.xtrain[i] = np.expand_dims(self.xtrain[i], axis=2)
            self.xtest[i] = np.array(self.xtest[i])
            self.xtest[i] = np.expand_dims(self.xtest[i], axis=2)
            #convert to one hot
            self.ytrain[i] = to_categorical(self.ytrain[i])
            self.ytest[i] = to_categorical(self.ytest[i])
        

    def build_seq_model(self, alpha = 0.5, filter = 100, kernel=15, activation=LeakyReLU(alpha=0.05), optimize='adam'):
        model = Sequential()
        n_samples, n_feats = self.xtrain[0].shape[1], self.xtrain[0].shape[2]
        #[[],[],[]]
        #add model layers
        model.add(Conv1D(filter, kernel_size=kernel, input_shape=(n_samples, n_feats)))
        model.add(activation)
        model.add(Conv1D(filter, kernel_size=kernel))
        model.add(activation)
        model.add(MaxPooling1D(pool_size=3))
        
        model.add(Conv1D(int(filter/2), kernel_size=int(kernel / 2)))
        model.add(activation)
        
        model.add(Conv1D(int(filter/2), kernel_size=int(kernel / 2)))
        model.add(activation)
        #model.add(GlobalAveragePooling1D())
        model.add(MaxPooling1D(pool_size=3))
        
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(50))
        model.add(activation)
        model.add(Dense(2, activation='softmax'))

        #optimzer 
        model.compile(optimizer=optimize, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model


    def run_model(self):
        #cross fold data
        self.pre_process()
        #get hyper parameters
        alpha, filters, kernel, optimizers = self.hypertune_params()

        #run every possible combination
        for a in alpha:
            for f in filters:
                for k in kernel:
                    for o in optimizers:
                        print("kernel ", k)
                        model = self.build_seq_model(alpha = a, filter = int(f), kernel = int(k), optimize = o)
                        
                        for i in range(len(self.xtrain)):
                            model.fit(self.xtrain[i], self.ytrain[i], epochs=3)
                            #validation_data=(self.xtest[i], self.ytest[i])
                            _, accuracy = model.evaluate(self.xtest[i], self.ytest[i])
                            #if accuracy is greater than 80 percent write configuration to file
                            print("acc ", accuracy)
                            if accuracy > .80:
                                with open("cnn_accuracy.txt", 'a+') as text:
                                    line = str(accuracy) + " Config alpha " + str(a) + " Filters " + str(f) + " Kernel " + str(k) + " optimizer " + str(optimizers.__class__)
                                    text.write(line)

        
    
    def seq_signal_data(self, x, y, timestep=120):
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



    def hypertune_params(self):
        #alpha values
        alpha = np.linspace(0.01, .09, num=9)
        print("alpha ", alpha)

        #filters
        Filters = np.linspace(20, 150, num=30, dtype=int)
        print("Filters ", Filters)

        #size of kernels
        Kernel_size = np.linspace(2, 40, num=38, dtype=int)
        print("Kernel ", Kernel_size)

        #learning rate
        learning_rate = np.linspace(.001, .010, num=10)

        #optimizer
        optimizers = [SGD(), RMSprop(), Adagrad(), Adadelta(), Adamax(), Nadam()]
        for i in range(len(learning_rate)):
            optimizers.append(SGD(lr=learning_rate[i]))
            optimizers.append(RMSprop(lr=learning_rate[i]))
            optimizers.append(Adagrad(lr=learning_rate[i]))
            optimizers.append(Adamax(lr=learning_rate[i]))
            optimizers.append(Nadam(lr=learning_rate[i]))

        return alpha, Filters, Kernel_size, optimizers



c = createCNN(2, 3, 3)
c.hypertune_params()