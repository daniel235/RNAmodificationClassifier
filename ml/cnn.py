from keras.models import Sequential
from keras import callbacks
from keras.layers import Dense, Conv1D, Conv2D, Flatten, Dropout, MaxPooling1D, LeakyReLU, GlobalAveragePooling1D, ELU
from keras.activations import elu, selu, tanh, sigmoid, hard_sigmoid, linear, relu
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam, Nadam, SGD, RMSprop, Adagrad, Adadelta, Adamax
import numpy as np
import sys


import crossFold as cfold

sys.path.insert(1, "../stats/")
import stats.stats as stats
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
            #self.xtest[i], self.ytest[i] = cfold.getEvenTestData(self.xtest[i], self.ytest[i])
            #self.xtest[i] = np.array(self.xtest[i])
            #self.xtrain[i] = np.expand_dims(self.xtrain[i], axis=2)
            #self.xtest[i] = np.expand_dims(self.xtest[i], axis=2)
            #reshape x data
            self.xtrain[i] = self.xtrain[i].reshape((self.xtrain[i].shape[0], self.xtrain[i].shape[1], 1))
            self.xtest[i] = self.xtest[i].reshape((self.xtest[i].shape[0], self.xtest[i].shape[1], 1))
            #convert to one hot
            self.ytrain[i] = to_categorical(self.ytrain[i])
            self.ytest[i] = to_categorical(self.ytest[i])
        

    def build_seq_model(self, alpha = 0.5, filter = 100, kernel=15, activator='relu', optimize='adam'):
        tensor_callback = callbacks.TensorBoard(log_dir="./logs/fit/tensorboard")
        model = Sequential()
        n_samples, n_feats = self.xtrain[0].shape[0], self.xtrain[0].shape[1]
        #hard code filters
        filter_size = n_feats / 2
        #[[],[],[]]
        #add model layers
        model.add(Conv1D(30, kernel_size=10, input_shape=(n_feats, 1), activation=activator))
        #model.add(LeakyReLU(alpha=0.5))
        model.add(Conv1D(30, kernel_size=10, activation=activator))
        #model.add(LeakyReLU(alpha=0.5))
        model.add(MaxPooling1D(pool_size=3))
        
        model.add(Conv1D(32, kernel_size=5, activation=activator))
        #model.add(LeakyReLU(alpha=0.5))
        
        model.add(Conv1D(32, kernel_size=2))
        model.add(LeakyReLU(alpha=0.5))
        #model.add(GlobalAveragePooling1D())
        
        model.add(MaxPooling1D(pool_size=3))
        
        model.add(Dropout(0.5))

        model.add(Flatten())
        #model.add(Dense(50, activation=activator))
        #model.add(LeakyReLU(alpha=0.5))
        model.add(Dense(2, activation='softmax'))

        #optimzer 
        model.compile(optimizer=optimize, loss='categorical_crossentropy', metrics=['accuracy'])
        print("Filter ", filter, " kernel ", kernel, " activation ", activator, " optimizer ", optimize)
        return model


    def run_model(self):
        #cross fold data
        self.pre_process()
        #get hyper parameters
        alpha, filters, kernel, optimizers, activation, progress = self.hypertune_params()
        percent = 0

        #run every possible combination
        for a in activation:
            for o in optimizers:
                for k in kernel:
                    for f in filters:
                        print("kernel ", k)
                        model = self.build_seq_model(filter = int(f), kernel = int(k), optimize = o, activator=a)
                        percent += 1
                        print("Percent ", float(percent / progress))
                        for i in range(len(self.xtrain)):
                            model.fit(self.xtrain[i], self.ytrain[i], epochs=3)
                            #validation_data=(self.xtest[i], self.ytest[i])
                            _, accuracy = model.evaluate(self.xtest[i], self.ytest[i])
                            #if accuracy is greater than 80 percent write configuration to file
                            print("acc ", accuracy)
    
                            if accuracy > .80:
                                with open("cnn_accuracy.txt", 'a+') as text:
                                    line = str(accuracy) + " Config alpha " + str(0.5) + " Filters " + str(f) + " Kernel " + str(k) + " optimizer " + str(optimizers.__class__) + "\n"
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
        alpha = np.linspace(0.01, .09, num=3)
        print("alpha ", alpha)

        #filters
        Filters = np.linspace(20, 150, num=30, dtype=int)
        print("Filters ", Filters)

        #size of kernels
        Kernel_size = np.linspace(2, 40, num=38, dtype=int)
        
        print("Kernel ", Kernel_size)

        #learning rate
        learning_rate = np.linspace(.001, .010, num=5)

        #optimizer
        optimizers = [SGD(), RMSprop(), Adagrad(), Adadelta(), Adamax(), Nadam()]
        '''
        for i in range(len(learning_rate)):
            optimizers.append(SGD(lr=learning_rate[i]))
            optimizers.append(RMSprop(lr=learning_rate[i]))
            optimizers.append(Adagrad(lr=learning_rate[i]))
            optimizers.append(Adadelta(lr=learning_rate[i]))
            optimizers.append(Adamax(lr=learning_rate[i]))
            optimizers.append(Nadam(lr=learning_rate[i]))
        '''
        #activation
        #activations = [relu(), elu(), selu(), tanh(), sigmoid(), hard_sigmoid(), linear()]
        activations = ['elu', 'selu', 'tanh', 'sigmoid', 'relu', 'hard_sigmoid', 'linear']

        progress = 30 * 38 * 10 * 6 * 7 * 10

        return alpha, Filters, Kernel_size, optimizers, activations, progress


    def single_run(self, f=75, a='sigmoid', k=15, optimizers=adam(lr=0.1)):
        self.pre_process()
        model = self.build_seq_model(filter=f, kernel=k, activator=a, optimize=optimizers)

        for i in range(len(self.xtrain)):
            tensor_callback = callbacks.TensorBoard(log_dir="./ml/logs/fit/tensorboard")
            hist = model.fit(self.xtrain[i], self.ytrain[i], epochs=5, callbacks=[tensor_callback], validation_split=0.2)
            #validation_data=(self.xtest[i], self.ytest[i])
            _, accuracy = model.evaluate(self.xtest[i], self.ytest[i])
            #if accuracy is greater than 80 percent write configuration to file
            print("acc ", accuracy)
            stats.signal_length_score(len(self.xtrain[i][0]), accuracy, f, k)

            if accuracy > .76:
                with open("cnn_accuracy.txt", 'a+') as text:
                    line = str(accuracy) + " Config alpha " + str(0.5) + " Filters " + str(f) + " Kernel " + str(k) + " optimizer " + str(optimizers.__class__) + "\n"
                    text.write(line)

        return hist
