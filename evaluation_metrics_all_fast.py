import pickle, numpy as np
from config import MODEL_PATH, K

model = pickle.load(open(MODEL_PATH, "rb"))
classes = model['classes']
n_classes = len(classes)

def load_features(feature_file):
    X, y, cls = pickle.load(open(feature_file, "rb"))
    return X, y

def knn_fast(X_train, y_train, x_test, k):
    distances = np.linalg.norm(X_train - x_test, axis=1)
    k_idx = np.argsort(distances)[:k]
    k_labels = y_train[k_idx]
    return np.bincount(k_labels).argmax()

def confusion_matrix(true_labels, pred_labels, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    return cm

def compute_metrics(cm):
    precisions, recalls, f1s = [], [], []

    accuracy = np.trace(cm) / np.sum(cm)

    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return accuracy, np.mean(precisions), np.mean(recalls), np.mean(f1s)

def evaluate(feature_file, name):
    X, y = load_features(feature_file)

    X = (X - model['mean']) / (model['std'] + 1e-8)

    y_true = []
    y_pred = []

    for i in range(len(X)):
        pred = knn_fast(model['X'], model['y'], X[i], K)
        y_true.append(y[i])
        y_pred.append(pred)

    cm = confusion_matrix(y_true, y_pred, n_classes)
    acc, prec, rec, f1 = compute_metrics(cm)

    print(f"\n[{name}]")
    print("-" * 60)
    print(f"Jumlah Data : {len(X)}")
    print(f"Accuracy    : {acc*100:.2f}%")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1-Score    : {f1:.4f}")

print("\n" + "="*70)
print("{:^70}".format("HASIL EVALUASI METRIK KLASIFIKASI SISTEM (FAST VERSION)"))
print("="*70)

evaluate("features_fasttrain.pkl", "TRAIN")
evaluate("features_fastvalidation.pkl", "VALIDATION")
evaluate("features_fasttest.pkl", "TEST")

print("\n" + "="*70)
print("Evaluasi selesai.")
print("="*70)
