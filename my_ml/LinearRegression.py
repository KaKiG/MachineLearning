import numpy as np


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        # combine the training data with "1" array
        X = np.hstack([np.ones((len(X_train), 1)), X_train])

        # based on the closed form expression, compute the result directly
        self._theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_train)

        # set coeffs to variable
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    # gradient descent method
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        # define the loss function
        def J(theta, X, y):
            try:
                return np.sum((y - X.dot(theta)) ** 2) / len(y)
            except:
                return float("inf")

        # define the derivative of loss function
        def dJ(theta, X, y):
            return X.T.dot(X.dot(theta) - y) * 2.0 / len(X)

        # define the gradient_descent method
        def gradient_descent(X, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0
            # iterate the gradient_descent in certain amount
            # or if the error is smaller than given
            while cur_iter < n_iters:
                gradient = dJ(theta, X, y)
                last_theta = theta
                theta = theta - eta * gradient

                if abs(J(last_theta, X, y) - J(theta, X, y)) <= epsilon:
                    break
            return theta

        # deal with data
        X = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros((X.shape[1], 1))

        # fit in gradient_descent
        self._theta = gradient_descent(X, y_train, initial_theta, eta, n_iters,)

        # set coeffs
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    # stochastic gradient_descent

    def fit_sgd(self, X_train, y_train, eta=0.01, n_iters=5):

        def dJ(theta, X, y):
            return X * (X.dot(theta) - y) * 2.


        def gradient_descent(X, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0
            while cur_iter < n_iters:
                # make sure every sample is used in a run
                index = np.random.permutation(len(X))
                X_new = X[index]
                y_new = y[index]
                for i in range(len(X)):
                    gradient = dJ(theta, X_new[i], y_new[i])
                    last_theta = theta
                    theta = theta - eta * gradient

                cur_iter += 1
            return theta

        X = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros((X.shape[1], 1))
        self._theta = gradient_descent(X, y_train, initial_theta, eta, n_iters,)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def _repr_(self):
        return "LinearRegression()"
