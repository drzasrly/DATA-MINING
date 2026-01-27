import pickle
import numpy as np
from config import MODEL_PATH, K

# ================= LOAD MODEL =================

model = pickle.load(open(MODEL_PATH, "rb"))
classes = model["classes"]
n_classes = len(classes)

print("\nModel berhasil dimuat")
print("Jumlah kelas :", n_classes)

# ================= LOAD FEATURE FILE =================

def load_features(feature_file):
    X, y, cls = pickle.load(open(feature_file, "rb"))
    return X, y

# ================= KNN FAST =================

def knn_fast(X_train, y_train, x_test, k):
    distances = np.linalg.norm(X_train - x_test, axis=1)
    k_idx = np.argsort(distances)[:k]
    k_labels = y_train[k_idx]
    return np.bincount(k_labels).argmax()

# ================= CONFUSION MATRIX =================

def confusion_matrix_multiclass(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# ================= ANALISIS TP FP FN =================

def analyze_cm(cm, classes):

    print("\n" + "="*120)
    print("ANALISIS CONFUSION MATRIX PER KELAS")
    print("="*120)

    print("{:25s} {:>6s} {:>6s} {:>6s} {:>10s} {:>10s}".format(
        "Kelas", "TP", "FN", "FP", "Precision", "Recall"
    ))
    print("-"*120)

    for i, cls in enumerate(classes):

        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

        print("{:25s} {:6d} {:6d} {:6d} {:10.4f} {:10.4f}".format(
            cls, tp, fn, fp, precision, recall
        ))

# ================= EVALUASI PER DATASET =================

def evaluate_confusion(feature_file, name):

    print("\n" + "="*120)
    print(f"CONFUSION MATRIX DATASET : {name}")
    print("="*120)

    X, y = load_features(feature_file)

    # Normalisasi
    X = (X - model["mean"]) / (model["std"] + 1e-8)

    y_true = []
    y_pred = []

    for i in range(len(X)):
        pred = knn_fast(model["X"], model["y"], X[i], K)
        y_true.append(y[i])
        y_pred.append(pred)

    # Confusion matrix
    cm = confusion_matrix_multiclass(y_true, y_pred, n_classes)

    # Cetak tabel confusion matrix
    print("\nCONFUSION MATRIX (Aktual x Prediksi)")
    print("-"*120)

    print("{:25s}".format("Aktual \\ Prediksi"), end="")
    for cls in classes:
        print("{:12s}".format(cls[:10]), end="")
    print()

    for i, cls in enumerate(classes):
        print("{:25s}".format(cls), end="")
        for j in range(n_classes):
            print("{:12d}".format(cm[i, j]), end="")
        print()

    # Analisis TP FP FN
    analyze_cm(cm, classes)


# ================= MAIN =================

evaluate_confusion("features_fasttrain.pkl", "TRAIN")
evaluate_confusion("features_fastvalidation.pkl", "VALIDATION")
evaluate_confusion("features_fasttest.pkl", "TEST")

print("\n" + "="*120)
print("Evaluasi confusion matrix semua dataset selesai.")
print("="*120)
