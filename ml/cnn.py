from keras.models import Sequential
from keras import callbacks
from keras.layers import Dense, Conv1D, Conv2D, Flatten, Dropout, MaxPooling1D, LeakyReLU, GlobalAveragePooling1D, ELU
from keras.activations import elu, selu, tanh, sigmoid, hard_sigmoid, linear, relu
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam, Nadam, SGD, RMSprop, Adagrad, Adadelta, Adamax
import numpy as np
import sys


import crossFold as cfold
import ml.pelican_CNN as pelican

sys.path.insert(1, "../stats/")
sys.path.insert(1, "../testing/")
import stats.stats as stats
import testing.learningCurve as lc
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
            #self.xtrain[i] = self.xtrain[i].reshape((1, self.xtrain[i].shape[0], self.xtrain[i].shape[1]))
            #self.xtest[i] = self.xtest[i].reshape((1, self.xtest[i].shape[0], self.xtest[i].shape[1]))
            #convert to one hot
            self.ytrain[i] = to_categorical(self.ytrain[i])
            self.ytest[i] = to_categorical(self.ytest[i])
        

    def build_seq_model(self, alpha = 0.5, filter = 100, kernel=15, activator='relu', optimize=Nadam(lr=0.02)):
        tensor_callback = callbacks.TensorBoard(log_dir="./logs/fit/tensorboard")
        model = Sequential()
        n_samples, n_feats = self.xtrain[0].shape[1], self.xtrain[0].shape[2]
        print(n_samples, n_feats)
        #hard code filters
        filter_size = int(n_feats / 2)
    
        #add model layers
        model.add(Conv1D(90, kernel_size=10, input_shape=(n_samples, n_feats), activation=activator))
        #model.add(LeakyReLU(alpha=0.5))
        model.add(Conv1D(90, kernel_size=10, activation=activator))
        #model.add(LeakyReLU(alpha=0.5))
        model.add(MaxPooling1D(pool_size=3))
        
        model.add(Conv1D(int(filter_size / 2), kernel_size=3, activation=activator))
        #model.add(LeakyReLU(alpha=0.5))
        '''
        model.add(Conv1D(int(filter_size / 2), kernel_size=3))
        model.add(LeakyReLU(alpha=0.5))
        #model.add(GlobalAveragePooling1D())
        
        model.add(MaxPooling1D(pool_size=3))
        '''
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


    def single_run(self, f=75, a='sigmoid', k=15, optimizers=adam(lr=0.1), pelican_run=True):
        self.pre_process()
        if pelican_run:
            sc = []
            for i in range(len(self.xtrain)):
                #shuffle xtrain and xtest
                idx = np.random.choice(len(self.xtrain[i]), len(self.xtrain[i]), replace=False)
                t_idx = np.random.choice(len(self.xtest[i]), len(self.xtest[i]), replace=False)
                self.xtrain[i], self.ytrain[i] = self.xtrain[i][idx], self.ytrain[i][idx]
                self.xtest[i], self.ytest[i] = self.xtest[i][t_idx], self.ytest[i][t_idx]
                sc = pelican.predictLabel(self.xtrain[i], 64, self.ytrain[i], self.xtest[i], self.ytest[i], score=sc)

            #plot real learning curves
            lc.createLearningCurve(None, None, None, None, name="Pelican", is_estimator=False, scores=sc)
            return

        model = self.build_seq_model(filter=f, kernel=k, activator=a, optimize=optimizers)

        for i in range(len(self.xtrain)):
            tensor_callback = callbacks.TensorBoard(log_dir="./ml/logs/fit/tensorboard")
            #shuffle data
            rand_index = np.random.choice(len(self.ytrain[i]), len(self.ytrain[i]), replace=False)
            test_rand_index = np.random.choice(len(self.ytest[i]), len(self.ytest[i]), replace=False)
            
            #shuffle data
            #self.xtrain[i] = self.xtrain[i][rand_index]
            #self.ytrain[i] = self.ytrain[i][rand_index]
            #self.xtest[i] = self.xtest[i][test_rand_index]
            #self.ytest[i] = self.ytest[i][test_rand_index]
            print("xtrain shape ", self.xtrain[i].shape)
            hist = model.fit(self.xtrain[i], self.ytrain[i], epochs=5, callbacks=[tensor_callback], validation_split=0.2)
            #validation_data=(self.xtest[i], self.ytest[i])
            #self.xtest[i], self.ytest[i] = cfold.getEvenTestData(self.xtest[i], self.ytest[i])
            _, accuracy = model.evaluate(self.xtest[i], self.ytest[i], verbose=1)
            #if accuracy is greater than 80 percent write configuration to file
            print("acc ", accuracy)
            stats.signal_length_score(len(self.xtrain[i][0]), accuracy, f, k)

            if accuracy > .76:
                with open("cnn_accuracy.txt", 'a+') as text:
                    line = str(accuracy) + " Config alpha " + str(0.5) + " Filters " + str(f) + " Kernel " + str(k) + " optimizer " + str(optimizers.__class__) + "\n"
                    text.write(line)

        return hist


    def ImageCNN(self):
        pass
