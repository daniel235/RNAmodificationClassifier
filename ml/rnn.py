import tensorflow.contrib.keras as keras

class createRNN():
    def __init__(self, nunits):
        self.network = keras.layers.LSTM(units=nunits)


    def hyperTune():
        #function to auto tune network 
        pass
