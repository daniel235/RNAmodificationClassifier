from keras import Sequential
from keras.layers import Dense
from keras.activations import tanh, sigmoid
from keras.optimizers import adam


def run_ann(x, y):
    xtrain, ytrain = x[:int(len(x) * .8)], y[:int(len(y) * .8)]
    xtest, ytest = x[int(len(x) * .8):], y[int(len(y) * .8):]
    
    model = Sequential()
    model.add(Dense(64, input_shape=(600,), activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    history = model.fit(xtrain, ytrain, verbose=0, epochs=50)

    _, accuracy = model.evaluate(xtest, ytest, verbose=1)
    print("accuracy ", accuracy * 100)
    