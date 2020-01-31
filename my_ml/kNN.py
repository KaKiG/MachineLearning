import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

class kNNClassifer:

    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, x_train, y_train):
        self._x_train = x_train
        self._y_train = y_train
        return self

    def _predict(self, x):
        distances = np.sqrt(np.sum((self._x_train-x)**2, axis = 1))
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def predict(self, X_predict):
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def score(self, X_test, y_test):
        return accuracy_score(y_test, self.predict(X_test))
