import numpy as np

def train_test_split(x, y, test_ratio = 0.2, seed=None):
    if seed != None:
        np.random.seed(seed)

    ind = np.random.permutation(x.shape[0])
    X_test = x[ind[:int(x.shape[0] * test_ratio)]]
    X_train = x[ind[int(x.shape[0] * test_ratio):]]
    y_test = y[ind[:int(y.shape[0] * test_ratio)]]
    y_train = y[ind[int(y.shape[0] * test_ratio):]]
    # train_combine = np.concatenate((x, y.reshape(-1,1)), axis = 1)
    # new_train = np.random.permutation(train_combine)
    # X_test = new_train[:int(new_train.shape[0] * test_ratio), :new_train.shape[1]-1]
    # X_train = new_train[int(new_train.shape[0] * test_ratio):, :new_train.shape[1]-1]
    # y_test = new_train[:int(new_train.shape[0] * test_ratio), new_train.shape[1]-1:]
    # y_train = new_train[int(new_train.shape[0] * test_ratio):, new_train.shape[1]-1:]

    return X_train, X_test, y_train, y_test
