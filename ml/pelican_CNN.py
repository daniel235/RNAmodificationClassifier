import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import platform
#import pandas as pd ### For future manipulations
#import scipy as sp ### For future manipulations
import matplotlib.pyplot as plt  #### Uncomment and use if you would like to see the traiing dataset length frequency plots
from sklearn.preprocessing import StandardScaler
# from sklearn import preprocessing, cross_validation, neighbors
#from sklearn.decomposition import PCA ### Uncomment if planning to do dimensionality reduction
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def predictLabel(X, bt, y_output, xtest, ytest, train=True, score=None):

    testX=X
    
    #print(trainX)
    #### Batch generator fucntion for dataset ########
    def batch(x, y, batch_size):
        scaler = StandardScaler()
        list_of_batches = []
        i=0
        while i<(len(x)-batch_size):
            start = i
            end = i+batch_size
            batch_x = np.array(x[start:end])
            batch_y = np.array(y[start:end])
            i += batch_size
            #print(len(batch_x))
            #scale batch
            scaler.fit(batch_x)
            batch_x = scaler.transform(batch_x)
            list_of_batches.append([batch_x, batch_y])

        return list_of_batches
    
    #### Random Batch generator fucntion for dataset ########
    
    def next_batch(num, data, labels):
        '''
        Return a total of `num` random samples and labels. 
        '''
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[ i] for i in idx]
        labels_shuffle = [labels[ i] for i in idx]
    
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    
    
    ######################### Parameters for Neural nets##########################
    tf.reset_default_graph()
    batch_size = bt# Batch size of dataset
    v_batch_size = bt# Batch size of dataset
    
    n_classes = 2 # Number of label classes
    n_steps=1 # Chunk size (1 dimension) 
    features=len(X[0]) # Feature size (length of signal-2 dimension)
    epsilon = 1e-3
     #input('\n'+'Save new model or Load from Previous checkpoint (type True for new & False for previous model): ')
    ######################### CNN Code ##########################
    x = tf.placeholder('float', shape=[None,features])
    y = tf.placeholder('float')
    
    keep_rate = 0.5
    keep_prob = tf.placeholder(tf.float32)
    
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
    def maxpool2d(x):
        #                        size of window         movement of window
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    
    ##### CNN function with 3 5*5 layers #####
    def convolutional_neural_network(x):
        weights = {'W_conv1':tf.Variable(tf.random_normal([1,5,1,32])),
                   'W_conv2':tf.Variable(tf.random_normal([1,5,32,64])),
                   'W_conv3':tf.Variable(tf.random_normal([1,5,64,128])),
                   'W_conv4':tf.Variable(tf.random_normal([1,5,128,256])),
                  
                   'W_fc':tf.Variable(tf.random_normal([9600,1024])),
                   'out':tf.Variable(tf.random_normal([1024, n_classes]))}
    
        biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
                   'b_conv2':tf.Variable(tf.random_normal([64])),
                   'b_conv3':tf.Variable(tf.random_normal([128])),
                   'b_conv4':tf.Variable(tf.random_normal([256])),
                   
                   'b_fc':tf.Variable(tf.random_normal([1024])),
                   'out':tf.Variable(tf.random_normal([n_classes]))}
    
        x = tf.reshape(x, shape=[-1, n_steps, features, 1])
    
        conv1 = tf.nn.leaky_relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = maxpool2d(conv1)
        conv1 = tf.nn.dropout(conv1, 0.1)
    #    print(conv1.shape) ## uncomment to check the shape of 1st CNN layer
        
        conv2 = tf.nn.leaky_relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = maxpool2d(conv2)
    #    print(conv2.shape) ## uncomment to check the shape of 1st CNN layer
        #conv2 = tf.nn.dropout(conv2, 0.1)

        conv3 = tf.nn.leaky_relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
        #conv3 = tf.nn.dropout(conv3, 0.1)
        #conv3 = maxpool2d(conv3)
    #    print(conv3.shape) ## uncomment to check the shape of 1st CNN layer
    
        conv4 = tf.nn.leaky_relu(conv2d(conv3, weights['W_conv4']) + biases['b_conv4'])
        #conv4 = tf.nn.dropout(conv4, 0.1)
        #conv2 == tf.contrib.layers.flatten(conv2)
        #print(conv2.shape)
        conv3s = conv4.get_shape().as_list()
        fc = tf.reshape(conv4,[-1, conv3s[1]*conv3s[2]*conv3s[3]])
        print(fc.shape)
        #fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
        fc = tf.nn.relu(fc)
        fc = tf.nn.dropout(fc, keep_rate)
    #    print(fc.shape)
        weights['out'] = tf.Variable(tf.random_normal([fc.get_shape().as_list()[1], n_classes]))
        output = tf.matmul(fc, weights['out'])+biases['out']
    #    print(output.shape) ## uncomment to print shape of output
        return output
        
    train_sess=True # change to False if you wanna load from previous checkpoint
    
    loss_v=[] 
    accu=[]
    label=[]


    def train_neural_network(x):
        with tf.name_scope('Model1'):
            prediction = convolutional_neural_network(x)
        #print(prediction)
        with tf.name_scope('Loss'):
            cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
        
        with tf.name_scope('OPtimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(cost)
            #optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.001, use_nesterov=True).minimize(cost)
    
        with tf.name_scope('Accuracy'):
        # # Accuracy
            acc = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            # predict = tf.argmax(prediction, 1)
    
        hm_epochs = 100 ### Epoch (time steps)
    
        saver = tf.train.Saver()
    
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", cost)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("accuracy", acc)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
    
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            #test_X=testX.reshape(v_batch_size,features)
    #               print(prediction)
            #preds=tf.nn.softmax(prediction)
#               print("Label:" + str(preds.eval(feed_dict={x: test_X})))
        
            if train:
                batches = batch(testX, y_output, v_batch_size)
                test_batches = batch(xtest, ytest, v_batch_size)
                loss_list = []
                lc_acc_train_x, lc_acc_train_y = [], []
                lc_acc_test_x, lc_acc_test_y = [], []
                for epoch in range(1000):
                    accuracy = 0
                    for _ in range(len(batches)):
                        test_X, y_batch = batches[_][0], batches[_][1]
                        #train first
                        op, loss = sess.run((optimizer, cost), feed_dict={x: test_X, y: y_batch})
                        loss_list.append(loss)
                        #get test data
                        accuracy += acc.eval(feed_dict={x: test_X, y: y_batch.reshape(v_batch_size,2)})
                        

                    lc_acc_train_y.append(accuracy / len(batches))
                    print(epoch, " epoch train ", (accuracy/len(batches)))
                    
                    accuracy = 0
                    for _ in range(len(test_batches)):
                        test_X, y_batch = test_batches[_][0], test_batches[_][1]
                        accuracy += acc.eval(feed_dict={x: test_X, y: y_batch})


                    lc_acc_test_y.append(accuracy / len(test_batches))
                    lc_acc_test_x.append(epoch)
                    print(epoch, " epoch test acc ", (accuracy/ len(test_batches)))
                    

                #stabilize learning curves
                lc_acc_test_x = np.linspace(0, 500, num=20).tolist()
                train_y_val = []
                test_y_val = []
                for index in lc_acc_test_x:
                    train_y_val.append(lc_acc_train_y[int(index)])
                    test_y_val.append(lc_acc_test_y[int(index)])

                plt.plot(lc_acc_test_x, train_y_val, label="training")
                plt.plot(lc_acc_test_x, test_y_val, label="validation")
                    
                
                plt.legend()
                plt.savefig("./results/PelicanLC.png")
                if platform.system() == "Windows":
                    plt.show()
                
                #add to scores
                #score.append([lc_acc_train_x, lc_acc_train_y, lc_acc_test_x, lc_acc_test_y])

            else:
                saver.restore(sess,dir +'/')
                print('Loaded latest check point....')
    #           for _ in range(int(len(testX)/v_batch_size)):
    #           test_X = btch(testX,v_batch_size)
                labs=str(sess.run(tf.argmax(prediction, 1), feed_dict={x: test_X, y: y_output}))
#               file5.write(labs+'\n')
            #accur=tf.argmax(prediction, 1)
                
    
    
    train_neural_network(x)
#    print(label)
    return score
