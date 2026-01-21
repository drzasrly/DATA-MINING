import os, cv2, pickle
import numpy as np
from feature_manual import extract_features
from knn_manual import knn_predict
from config import MODEL_PATH, K, IMG_SIZE

# =====================================================
# LOAD MODEL
# =====================================================
model = pickle.load(open(MODEL_PATH, "rb"))
classes = model['classes']
n_classes = len(classes)

# =====================================================
# FUNGSI EVALUASI DATASET + CONFUSION MATRIX
# =====================================================
def evaluate_dataset(dataset_dir):
    y_true = []
    y_pred = []

    for cls in os.listdir(dataset_dir):
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Preprocess
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            feat = np.array(extract_features(img))
            feat = (feat - model['mean']) / (model['std'] + 1e-8)

            # Predict
            pred = knn_predict(model['X'], model['y'], feat, K)
            pred_label = model['classes'][pred]

            y_true.append(cls)
            y_pred.append(pred_label)

    return y_true, y_pred


# =====================================================
# HITUNG CONFUSION MATRIX MANUAL
# =====================================================
def confusion_matrix(true_labels, pred_labels, classes):
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for t, p in zip(true_labels, pred_labels):
        i = classes.index(t)
        j = classes.index(p)
        cm[i, j] += 1

    return cm


# =====================================================
# HITUNG METRIK DARI CONFUSION MATRIX
# =====================================================
def compute_metrics(cm):
    precisions = []
    recalls = []
    f1s = []

    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    accuracy = total_correct / total_samples if total_samples > 0 else 0

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

    # Macro average
    precision_macro = np.mean(precisions)
    recall_macro    = np.mean(recalls)
    f1_macro        = np.mean(f1s)

    return accuracy, precision_macro, recall_macro, f1_macro


# =====================================================
# EVALUASI SEMUA SPLIT
# =====================================================
splits = {
    "TRAIN"       : "tomato_disease_ready/train",
    "VALIDATION" : "tomato_disease_ready/validation",
    "TEST"        : "tomato_disease_ready/test"
}

print("\n" + "="*70)
print("{:^70}".format("HASIL EVALUASI METRIK KLASIFIKASI SISTEM"))
print("="*70)

for name, path in splits.items():
    y_true, y_pred = evaluate_dataset(path)
    cm = confusion_matrix(y_true, y_pred, classes)
    acc, prec, rec, f1 = compute_metrics(cm)

    print(f"\n[{name}]")
    print("-" * 60)
    print(f"Jumlah Data : {len(y_true)}")
    print(f"Accuracy    : {acc*100:.2f}%")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1-Score    : {f1:.4f}")

print("\n" + "="*70)
print("Evaluasi selesai.")
print("="*70)
