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
        print("i shape", inputs.shape)
        #unroll y output
        currentY = []
        for i in range(len(y_output)):
            currentY.append(y_output[i][0])

        y_output = np.array(currentY)

        #inputs = inputs.reshape((1, inputs.shape[0], inputs.shape[1]))
        n_samples, n_features = inputs[0].shape[0], inputs[0].shape[1]
        n_neurons = 20

        #720 x 100
        x = tf.placeholder(tf.float32, [None, n_samples, n_features])
        y = tf.placeholder(tf.int32, [None])
        #dynamic rnn
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

        #uneven sequence length
        #seq_length = tf.placeholder(tf.int32, [None])

        outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)

        #states = tf.reshape(states, [n_samples, None])
        #logits = tf.layers.dense(states, 2)
        #outputs = tf.contrib.layers.flatten(outputs)
        logits = tf.contrib.layers.fully_connected(outputs, 2)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=np.transpose(y_output), logits=logits)
        
        '''
        mask = tf.sign(tf.reduce_max(tf.abs(y_output), 2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, 1)
        cross_entropy /= tf.reduce_sum(mask, 1)
        '''
        prediction = tf.nn.softmax(outputs)
        cost = -tf.reduce_mean(cross_entropy)

        #optimize network
        optimize = tf.train.AdamOptimizer()
        optOperation = optimize.minimize(cost)

        #get accuracy 
        #correct = tf.nn.in_top_k(logits, y_output, 1)
        #accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            sess.run(optOperation, feed_dict={x: inputs, y: y_output})
            #prediction = sess.run(prediction, feed_dict={x: inputs, y: y_output})
            #print(prediction)
            #acc_train = accuracy.eval(feed_dict={x: inputs, y: y_output})


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
        print("batches shape ", batches.shape)
        for i in range(len(batches)):
            model.fit(batches[i], batches_out[i], epochs=150, validation_data=(self.xtest[i], self.ytest[i]))

        for i in range(len(testBatch)):
            _, accuracy = model.evaluate(testBatch[i], testBatchOut[i])
            print("acc ", accuracy)
    
    def createBatchData(self):
        batch_size = 32
        batches = []
        batch = []
        batches_out = []
        batch_y = []

        #test batches
        testBatches = []
        testBatchOutput = []
        testBatch = []
        testBatchY = []

        #create batches to run rnn
        for i in range(len(self.xtrain)):
            for j in range(len(self.xtrain[i])):
                if j % 32 == 0 and j != 0:
                    batches.append(np.array(batch))
                    batches_out.append(np.array(batch_y))
                    batch = []
                    batch_y = []

                batch.append(self.xtrain[i][j])
                batch_y.append(self.ytrain[i][j])
                #self.xtrain[i] = self.xtrain[i].reshape(self.xtrain[i].shape[0], self.xtrain[i].shape[1])
                #self.xtest[i] = self.xtest[i].reshape(self.xtest[i].shape[0], self.xtest[i].shape[1])

            for j in range(len(self.xtest[i])):
                if j % 32 == 0 and j != 0:
                    testBatches.append(np.array(testBatch))
                    testBatchOutput.append(np.array(testBatchY))
                    testBatch = []
                    testBatchY = []
                
                testBatch.append(self.xtest[i][j])
                testBatchY.append(self.ytest[i][j])


        return np.array(batches), np.array(batches_out), np.array(testBatches), np.array(testBatchOutput)