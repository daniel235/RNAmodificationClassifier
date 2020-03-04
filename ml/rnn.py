import tensorflow.contrib.keras as keras
import tensorflow as tf
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import LSTM, Dense, Dropout, Input, Flatten
import numpy as np
import ml.crossFold as cfold


class createRNN():
    def __init__(self, x, y):
        self.network = keras.layers.LSTM(units=10)
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None
        self.xtrain, self.ytrain, self.xtest, self.ytest = cfold.splitData(3, x, y)


    def hyperTune(self):
        #function to auto tune network 
        pass

    #todo scale inputs
    def runRecurrentNet(self, inputs, y_output):
        tf.reset_default_graph()
        #each signal is a time step
        #n_input should be padded length
        #inputs = np.array(inputs)
        
        #unroll y output
        currentY = []
        for i in range(len(y_output)):
            currentY.append(y_output[i])

        #y_output = np.array(currentY).reshape((y_output.shape[0] * y_output.shape[1], 2))
        
        #seq length
        n_steps = inputs.shape[1]
        #only 1 feature
        n_features = 1
        n_neurons = 30

        #720 x 100
        #nsteps is number of samples(32) features(1)
        x = tf.placeholder(tf.float32, [None, n_steps, n_features], name="X_input")
        y = tf.placeholder(tf.int32, [None])
        #dynamic rnn
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, name="rnn_cell_basic", activation=tf.nn.relu)
        lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, name="lstm_cell_basic")
        #uneven sequence length
        #seq_length = tf.placeholder(tf.int32, [None])

        outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)

        #states = tf.reshape(states, [n_samples, None])
        logits = tf.layers.dense(states, 2)
        #outputs = tf.contrib.layers.flatten(outputs)
        #logits = tf.contrib.layers.fully_connected(outputs, 2)
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_output, logits=logits)
        
        '''
        mask = tf.sign(tf.reduce_max(tf.abs(y_output), 2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, 1)
        cross_entropy /= tf.reduce_sum(mask, 1)
        '''
        prediction = tf.nn.softmax(logits)
        cost = tf.reduce_mean(cross_entropy)

        #optimize network
        optimize = tf.train.AdamOptimizer()
        optOperation = optimize.minimize(cost)
        #correct = tf.nn.in_top_k(logits, y, 1)
        
        #get accuracy 
        #correct = tf.nn.in_top_k(prediction, y_output, 1)
        #accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        init = tf.global_variables_initializer()
    

        with tf.Session() as sess:
            sess.run(init)
            #sess.run(optOperation, feed_dict={x: inputs, y: y_output})
            #prediction = sess.run(prediction, feed_dict={x: inputs, y: y_output})
            #print(prediction)
            #acc_train = accuracy.eval(feed_dict={x: inputs, y: y_output})
            for epoch in range(100):
                sess.run(optOperation, feed_dict={x: inputs})

            c = sess.run(prediction, feed_dict={x: inputs})
            #print("prediction ", c)
            accuracy = 0
            for num, pred in enumerate(c):
                print(num, pred, y_output[int(num)])
                true_val = np.argmax(pred, axis=0)
                if y_output[int(num)][true_val] == 1.0:
                    accuracy += 1

            accuracy = float(accuracy / len(y_output))
            print(accuracy)


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
    

    def runRNN(self, x, y):
        #for i in range(len(x)):
            #for batch in range(len(x[i])):
                #self.runRecurrentNet(np.expand_dims(x[i][batch], axis=2), y[i][batch])
        
        for i in range(len(self.xtrain)):
            idx = np.random.choice(len(self.xtrain[i]), len(self.xtrain[i]), replace=False)
            self.xtrain[i] = self.xtrain[i][idx]
            self.ytrain[i] = self.ytrain[i][idx]
            self.runRecurrentNet(np.expand_dims(self.xtrain[i], axis=2), to_categorical(self.ytrain[i]))

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