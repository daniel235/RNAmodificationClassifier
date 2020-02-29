import tensorflow.contrib.keras as keras
import tensorflow as tf
from keras.models import Sequential
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


    def runRecurrentNet(self, inputs, y_output):
        #each signal is a time step
        #n_input should be padded length
        #inputs = np.array(inputs)
        
        #unroll y output
        currentY = []
        for i in range(len(y_output)):
            currentY.append(y_output[i][0])

        y_output = np.array(currentY)

        #inputs = inputs.reshape((1, inputs.shape[0], inputs.shape[1]))
        n_steps = inputs.shape[0]
        n_features = inputs[0].shape[1]
        n_neurons = 10

        #720 x 100
        #nsteps is number of samples(32) features(109)
        x = tf.placeholder(tf.float32, [None, n_steps, n_features])
        y = tf.placeholder(tf.int32, [None])
        #dynamic rnn
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

        #uneven sequence length
        #seq_length = tf.placeholder(tf.int32, [None])

        outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)

        #states = tf.reshape(states, [n_samples, None])
        #logits = tf.layers.dense(states, 2)
        #outputs = tf.contrib.layers.flatten(outputs)
        #logits = tf.contrib.layers.fully_connected(outputs, 2)
        '''
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=np.transpose(y_output), logits=logits)
        
        
        mask = tf.sign(tf.reduce_max(tf.abs(y_output), 2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, 1)
        cross_entropy /= tf.reduce_sum(mask, 1)
        
        prediction = tf.nn.softmax(outputs)
        cost = -tf.reduce_mean(cross_entropy)

        #optimize network
        optimize = tf.train.AdamOptimizer()
        optOperation = optimize.minimize(cost)

        #get accuracy 
        #correct = tf.nn.in_top_k(logits, y_output, 1)
        #accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        '''
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            #sess.run(optOperation, feed_dict={x: inputs, y: y_output})
            #prediction = sess.run(prediction, feed_dict={x: inputs, y: y_output})
            #print(prediction)
            #acc_train = accuracy.eval(feed_dict={x: inputs, y: y_output})
            for i in range(len(inputs)):
                output = sess.run(outputs, feed_dict={x: inputs[i]})
                print(output)


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
    

    def createBatchData(self):
        cfoldBatches = []
        testcfoldBatches = []
        cfoldY = []
        ctestfoldY = []


        for i in range(len(self.xtrain)):
            #manually get shape of data then initialize zero numpy array
            batch_count = int(len(self.xtrain[i]) / 32)
            seq_length = len(self.xtrain[i][0])

            #test batch zero array
            test_batch_count = int(len(self.xtest[i]) / 32)

            #create zero array
            batch = np.zeros((batch_count, 32, seq_length))
            test_batch = np.zeros((test_batch_count, 32, seq_length))
            cfoldybatch = np.zeros((batch_count, 32, 1))
            ctestfoldYbatch = np.zeros((test_batch_count, 32, 1))

            cfoldBatches.append(batch)   
            testcfoldBatches.append(test_batch)
            cfoldY.append(cfoldybatch)
            ctestfoldY.append(ctestfoldYbatch)


        #create batches to run rnn
        for i in range(len(self.xtrain)):
            batch_index = 0
            for j in range(len(self.xtrain[i])):
                if j % 32 == 0 and j != 0:
                    batch_index += 1

                cfoldBatches[i][batch_index][j % 32] = self.xtrain[i][j]
                cfoldY[i][batch_index][j % 32] = self.ytrain[i][j]
                
            batch_index = 0
            for j in range(len(self.xtest[i])):
                if j % 32 == 0 and j != 0:
                    batch_index += 1
                
                testcfoldBatches[i][batch_index][j % 32] = self.xtest[i][j]
                ctestfoldY[i][batch_index][j % 32] = self.ytest[i][j]

        return cfoldBatches, cfoldY, testcfoldBatches, ctestfoldY