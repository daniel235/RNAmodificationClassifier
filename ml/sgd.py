import numpy as np
from sklearn.linear_model import SGDClassifier


def gdClassifier(x, y):
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(x, y)
    