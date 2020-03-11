import tensorflow.contrib.keras as keras
import tensorflow as tf
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import LSTM, Dense, Dropout, Input, Flatten
import matplotlib.pyplot as plt
import numpy as np
import ml.crossFold as cfold


class createRNN():
    def __init__(self, x, y):
        self.network = keras.layers.LSTM(units=10)
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None
        self.xtrain, self.ytrain, self.xtest, self.ytest = cfold.splitData(5, x, y)
        

    def hyperTune(self):
        #function to auto tune network 
        pass

    def getRNNCells(self, lstm=False, n_neurons=20, activate=tf.nn.leaky_relu):
        if lstm:
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=activate, state_is_tuple=False)
        else:
            cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=activate)
        
        return cell


    def getMultiLayer(self, layers=2, lstm=False, neurons=20, activate=tf.nn.leaky_relu):
        multi_cell = tf.contrib.rnn.MultiRNNCell([self.getRNNCells(lstm=True, n_neurons=neurons, activate=activate) for layer in range(layers)], state_is_tuple=False)
        return multi_cell


    #todo scale inputs
    def runRecurrentNet(self, inputs, y_output, xtest, ytest):
        tf.reset_default_graph()
        #each signal is a time step
        #n_input should be padded length
        #inputs = np.array(inputs)

        #y_output = np.array(currentY).reshape((y_output.shape[0] * y_output.shape[1], 2))
        
        #seq length
        n_steps = inputs.shape[1]
        #only 1 feature
        n_features = 1
        n_neurons = 25

        #720 x 100
        #nsteps is number of samples(32) features(1)
        x = tf.placeholder(tf.float32, [None, n_steps, n_features], name="X_input")
        y = tf.placeholder(tf.int32, [None, 2])
        #dynamic rnn
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, name="rnn_cell_basic", activation=tf.nn.leaky_relu)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, name="lstm_cell_basic", state_is_tuple=False)
        #uneven sequence length
        seq_length = tf.placeholder(tf.int32, [None])

        outputs, states = tf.nn.dynamic_rnn(self.getMultiLayer(layers=3, lstm=True), x, dtype=tf.float32, sequence_length=seq_length)
        #stat_outputs, stat_states = tf.nn.static_rnn(lstm_cell)
        #states = tf.reshape(states, [n_samples, None])
        logits = tf.layers.dense(states, 2)
        #outputs = tf.contrib.layers.flatten(outputs)
        #logits = tf.contrib.layers.fully_connected(outputs, 2)
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        
        '''
        mask = tf.sign(tf.reduce_max(tf.abs(y_output), 2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, 1)
        cross_entropy /= tf.reduce_sum(mask, 1)
        '''
        prediction = tf.nn.softmax(logits)
        cost = tf.reduce_mean(cross_entropy)

        #optimize network
        optimize = tf.train.AdamOptimizer().minimize(cost)
        #optimize = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
        
        #correct = tf.nn.in_top_k(logits, y, 1)
        
        #get accuracy 
        #correct = tf.nn.in_top_k(prediction, y_output, 1)
        #accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            #get variablesequence length 
            train_seq_len = self.getSequenceLength(inputs)
            test_seq_len = self.getSequenceLength(xtest)
            
            predictions = []
            accuracies = []
            accuracies_test = []
            epochs = []
            #train network
            for epoch in range(1000):
                _, preds = sess.run((optimize, prediction), feed_dict={x: inputs, y: y_output, seq_length: train_seq_len})
                
                accuracy = 0
                #iterate through predictions
                for num, pred in enumerate(preds):
                    true_val = np.argmax(pred, axis=0)
                    if y_output[int(num)][true_val] == 1.0:
                        accuracy += 1
                
                epochs.append(epoch)

                accuracies.append(float(accuracy / len(y_output)))
                #run test 
                c = sess.run(prediction, feed_dict={x: xtest, y: ytest, seq_length: test_seq_len})
                accuracy = 0
                
                #iterate through predictions
                for num, pred in enumerate(c):
                    true_val = np.argmax(pred, axis=0)
                    if ytest[int(num)][true_val] == 1.0:
                        accuracy += 1

                accuracies_test.append(float(accuracy / len(ytest)))
           
            #plot learning curves
            plt.plot(epochs, accuracies, label="training")
            plt.plot(epochs, accuracies_test, label="testing")
            plt.legend()
            plt.savefig("./results/RNNlc.png")
            plt.show()
            plt.close()


    def prepareRNN(self, x, y):
        #for i in range(len(x)):
            #for batch in range(len(x[i])):
                #self.runRecurrentNet(np.expand_dims(x[i][batch], axis=2), y[i][batch])
        
        for i in range(len(self.xtrain)):
            idx = np.random.choice(len(self.xtrain[i]), len(self.xtrain[i]), replace=False)
            self.xtrain[i] = self.xtrain[i][idx]
            self.ytrain[i] = self.ytrain[i][idx]
            idx = np.random.choice(len(self.xtest[i]), len(self.xtest[i]), replace=False)
            self.xtest[i] = self.xtest[i][idx]
            self.ytest[i] = self.ytest[i][idx]
            #self.xtest[i], self.ytest[i] = cfold.getEvenTestData(self.xtest[i], self.ytest[i])
            print("x ", len(self.xtest[i]), " y ", len(self.ytest[i]))
            self.runRecurrentNet(np.expand_dims(self.xtrain[i], axis=2), to_categorical(self.ytrain[i]), np.expand_dims(self.xtest[i], axis=2), to_categorical(self.ytest[i]))



    #input batch size, time_steps, data dim
    def createLSTM(self):
        for i in range(len(self.xtrain)):
            self.xtrain[i] = np.array(self.xtrain[i])


        n_samples, n_features = self.xtrain[0].shape[0], self.xtrain[0].shape[1]
        model = Sequential()

        #model.add(Input(shape=(n_samples, n_features)))
        model.add(LSTM(30, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, input_shape=(None, n_features)))

        #fully connected layer
        model.add(Dense(30, activation='relu'))

        #dropout 
        model.add(Dropout(0.5))
        model.summary()
        #output
        #model.add(Flatten())
        model.add(Dense(2, activation='softmax'))

        #compile
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    

    def runLSTM(self, model):
        batches, batches_out, testBatch, testBatchOut = self.createBatchData()
        batches.reshape((batches.shape[0], batches[0].shape[0]))
        print("batches shape ", batches.shape)
        model.fit(batches, batches_out, epochs=150, validation_data=(testBatch, testBatchOut))

        for i in range(len(testBatch)):
            _, accuracy = model.evaluate(testBatch[i], testBatchOut[i])
            print("acc ", accuracy)
    

    
    #todo shuffle data
    def createBatchData(self):
        cfoldBatches = []
        testcfoldBatches = []
        cfoldY = []
        ctestfoldY = []

        for i in range(len(self.xtrain)):
            #shuffle data
            idx = np.random.choice(len(self.xtrain[i]), len(self.xtrain[i]), replace=False)
            self.xtrain[i] = self.xtrain[i][idx]
            self.ytrain[i] = self.ytrain[i][idx]
            
            #manually get shape of data then initialize zero numpy array
            batch_count = int(len(self.xtrain[i]) / 64)
            seq_length = len(self.xtrain[i][0])

            #test batch zero array
            test_batch_count = int(len(self.xtest[i]) / 64)

            #create zero array
            batch = np.zeros((batch_count, 64, seq_length))
            test_batch = np.zeros((test_batch_count, 64, seq_length))
            cfoldybatch = np.zeros((batch_count, 64, 2))
            ctestfoldYbatch = np.zeros((test_batch_count, 64, 2))

            cfoldBatches.append(batch)   
            testcfoldBatches.append(test_batch)
            cfoldY.append(cfoldybatch)
            ctestfoldY.append(ctestfoldYbatch)


        #create batches to run rnn
        for i in range(len(self.xtrain)):
            batch_index = 0
            for j in range(len(self.xtrain[i])):
                if j % 64 == 0 and j != 0:
                    batch_index += 1

                if batch_index < cfoldBatches[i].shape[0]:
                    cfoldBatches[i][batch_index][j % 64] = self.xtrain[i][j]
                    cfoldY[i][batch_index][j % 64] = to_categorical(self.ytrain[i][j][0])
                    
            
            batch_index = 0
            for j in range(len(self.xtest[i])):
                if j % 64 == 0 and j != 0:
                    batch_index += 1
                
                if batch_index < testcfoldBatches[i].shape[0]:
                    testcfoldBatches[i][batch_index][j % 64] = self.xtest[i][j]
                    ctestfoldY[i][batch_index][j % 64] = to_categorical(self.ytest[i][j][0])


        return cfoldBatches, cfoldY, testcfoldBatches, ctestfoldY


    def createLearningCurve(self, c, y):
        print("len of y ", len(y))
        #create learning curve
        break_points = np.linspace(0, len(c), 5, endpoint=False, dtype=int)[1:]
        
        lin_acc_x, lin_acc_y = [], []
        accuracy = 0
        currentPoint = 0
        #iterate through predictions
        for num, pred in enumerate(c):
            true_val = np.argmax(pred, axis=0)
            if y[int(num)][true_val] == 1.0:
                accuracy += 1

            #plot breakpoint
            if currentPoint < len(break_points):
                if num == break_points[currentPoint]:
                    lin_acc_x.append(num)
                    lin_acc_y.append(float(accuracy / num))
                    currentPoint += 1

        accuracy = float(accuracy / len(y))
        lin_acc_x.append(len(y))
        lin_acc_y.append(accuracy)
        print(accuracy)
        #plot learning curve
        print("x ", lin_acc_x, " y ", lin_acc_y)
        plt.plot(lin_acc_x, lin_acc_y, label="training")
        plt.savefig("./results/rnnLC")
        plt.show()
        plt.close()


    #get true sequence length of individual signals
    def getSequenceLength(self, x):
        seq_len = []
        for i in range(len(x)):
            count = 0
            for j in range(len(x[i])):
                if count == (len(x[i]) - 1):
                    seq_len.append(count)
                    break

                if x[i][j] == 0:
                    seq_len.append(count)
                    break
                else:
                    count += 1

        return seq_len