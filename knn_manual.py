import numpy as np

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def knn_predict(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        distances.append((euclidean(X_train[i], x_test), y_train[i]))
    distances.sort(key=lambda x: x[0])
    k_labels = [label for _, label in distances[:k]]
    return max(set(k_labels), key=k_labels.count)