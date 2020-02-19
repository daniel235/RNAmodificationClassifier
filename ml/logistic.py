from sklearn.linear_model import LogisticRegression
import numpy as np

class logRegression():
    def __init__(self):
        self.reg = LogisticRegression(random_state=42, max_iter=1000000)


    def fit(self, x, y):
        train = x[:int(len(x) * .7)]
        train_output = y[:int(len(y) * .7)]
        test = x[int(len(x) * .7):]
        test_output = y[int(len(y) * .7):]
        self.reg.fit(train, train_output)
        predictions = self.reg.predict(test)
        accuracy = 0
        for i in range(len(test_output)):
            if test_output[i] == predictions[i]:
                accuracy += 1

        accuracy = accuracy / len(test)
        print(accuracy)
    
