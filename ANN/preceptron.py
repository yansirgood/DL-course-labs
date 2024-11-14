import numpy as np


class MyPerceptron:
    def __init__(self, l_rate=0.1):
        self.w = None
        self.b = 0
        self.l_rate = l_rate

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    # use SGD to train
    def fit(self, X_train, y_train):
        wrong = False
        if self.w is None:
            self.w = np.zeros(X_train.shape[1], dtype=np.float32)
        while not wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                wrong = True
        return 'Perceptron Model!'

    def score(self, X_test, y_test):
        correct = 0
        for d in range(len(X_test)):
            X = X_test[d]
            y = y_test[d]
            if y * self.sign(X, self.w, self.b) > 0:
                correct += 1
        return correct / len(X_test)
