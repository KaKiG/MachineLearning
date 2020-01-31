import numpy as np

def accuracy_score(y_true, y_predict):
    return np.sum(y_true == y_predict) / len(y_true)
