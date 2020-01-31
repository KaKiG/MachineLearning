import numpy as np


class PCA:

    def __init__(self, n):

        self.n_components = n
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):

        def demean(X):
            return X-np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w) ** 2), axis=0) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):
            w = initial_w
            w = direction(w)
            curr_iter = 0

            while curr_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if abs(f(w, X)-f(last_w, X)) < epsilon:
                    break

                curr_iter += 1
            return w

        X_PCA = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X_PCA.shape[1]))
        for i in range(self.n_components):
            initial_w = X_PCA[0, :]
            w = first_component(X_PCA, initial_w)
            self.components_[i, :] = w
            X_PCA = X_PCA - X_PCA.dot(w).reshape(-1, 1) * w

        return self
