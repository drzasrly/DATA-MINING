import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from feature_manual import extract_features
from knn_manual import knn_predict

DATASET = "tomato_disease_ready"

def load_data(split):
    X, y = [], []
    classes = sorted(os.listdir(os.path.join(DATASET, split)))

    for i, cls in enumerate(classes):
        for f in os.listdir(os.path.join(DATASET, split, cls)):
            img = cv2.imread(os.path.join(DATASET, split, cls, f))
            img = cv2.resize(img, (256,256))
            X.append(extract_features(img))
            y.append(i)
    return np.array(X), np.array(y)

# Load dataset
X_train, y_train = load_data("train")
X_val, y_val = load_data("validation")

# Normalisasi
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# Evaluasi K
K_values = [1,3,5,7,9]
train_acc, val_acc = [], []   

def accuracy(Xref, yref, Xeval, yeval, K):
    correct = 0
    for i in range(len(Xeval)):
        pred = knn_predict(Xref, yref, Xeval[i], K)
        if pred == yeval[i]:   
            correct += 1
    return correct / len(yeval) * 100

print("Evaluation")
print("-"*50)

for i, K in enumerate(K_values):
    acc_tr = accuracy(X_train, y_train, X_train, y_train, K)
    acc_vl = accuracy(X_train, y_train, X_val, y_val, K)

    train_acc.append(acc_tr)
    val_acc.append(acc_vl)

    print(f"Iteration {i+1} | K={K} | Train={acc_tr:.2f}% | Val={acc_vl:.2f}%")

# Plot
plt.figure()
plt.plot(K_values, train_acc, marker='o', label='Training')
plt.plot(K_values, val_acc, marker='o', label='Validation')
plt.xlabel("Nilai K")
plt.ylabel("Akurasi (%)")
plt.title("Evaluation KNN")
plt.legend()
plt.grid()
plt.show()
