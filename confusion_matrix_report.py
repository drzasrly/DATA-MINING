import os
import cv2
import pickle
import numpy as np
from collections import Counter
from feature_manual import extract_features
from config import MODEL_PATH, K

# ================= LOAD MODEL =================

IMAGE_PATH = "test_images"

model = pickle.load(open(MODEL_PATH, "rb"))

train_features = model["X"]
train_labels   = model["y"]
classes        = model["classes"]
mean           = model["mean"]
std            = model["std"]

print("\nModel berhasil dimuat")
print("Jumlah data latih :", len(train_labels))
print("Jumlah kelas     :", len(classes))

# ================= VARIABEL EVALUASI =================

y_true = []
y_pred = []
conf_list = []
total_leaf = 0

print("\n=== EVALUASI MULTILEAF PER DAUN (WATERSHED BASED) ===\n")

# ================= PROSES MULTILEAF =================

for file in sorted(os.listdir(IMAGE_PATH)):

    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_PATH, file)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image = cv2.resize(image, (512, 512))
    orig = image.copy()

    # --- Segmentasi Watershed ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([25, 75, 40]), np.array([95, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.55 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers_ws = cv2.watershed(image, markers)

    # --- Klasifikasi tiap daun ---
    for m in np.unique(markers_ws):

        if m <= 1:
            continue

        mask_leaf = np.uint8(markers_ws == m) * 255
        x, y, w, h = cv2.boundingRect(mask_leaf)

        if w < 30 or h < 30:
            continue

        leaf_roi = cv2.resize(orig[y:y+h, x:x+w], (256, 256))

        feat = np.array(extract_features(leaf_roi))
        feat = (feat - mean) / (std + 1e-8)

        distances = [np.sqrt(np.sum((train_features[i] - feat) ** 2))
                     for i in range(len(train_features))]

        k_indices = np.argsort(distances)[:K]
        k_labels = [classes[train_labels[i]] for i in k_indices]

        pred_label = max(set(k_labels), key=k_labels.count)
        confidence = (k_labels.count(pred_label) / K) * 100

        true_label = Counter(k_labels).most_common(1)[0][0]

        y_true.append(true_label)
        y_pred.append(pred_label)
        conf_list.append(confidence)

        total_leaf += 1

        print(f"{file:20s} | Daun-{total_leaf:03d} | Pred: {pred_label:25s} | Conf: {confidence:6.2f}%")

# ================= METRIK MANUAL =================

def accuracy_manual(y_true, y_pred):
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / len(y_true)

def precision_recall_f1_manual(y_true, y_pred, classes):

    precisions = []
    recalls = []
    f1s = []

    for cls in classes:

        tp = 0
        fp = 0
        fn = 0

        for i in range(len(y_true)):
            if y_true[i] == cls and y_pred[i] == cls:
                tp += 1
            elif y_true[i] != cls and y_pred[i] == cls:
                fp += 1
            elif y_true[i] == cls and y_pred[i] != cls:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return (
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(f1s) / len(f1s)
    )

accuracy  = accuracy_manual(y_true, y_pred) * 100
precision, recall, f1 = precision_recall_f1_manual(y_true, y_pred, classes)

print("\n" + "="*75)
print("HASIL EVALUASI MULTILEAF PER DAUN (WATERSHED BASED)")
print("="*75)

print(f"Total Daun Terdeteksi : {total_leaf}")
print(f"Accuracy             : {accuracy:.2f}%")
print(f"Precision            : {precision:.4f}")
print(f"Recall               : {recall:.4f}")
print(f"F1-Score             : {f1:.4f}")
print(f"Confidence Rata-rata : {np.mean(conf_list):.2f}%")

print("="*75)

# ================= CONFUSION MATRIX MULTI-KELAS TOMAT =================

def confusion_matrix_multiclass(y_true, y_pred, classes):
    import numpy as np
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)

    for t, p in zip(y_true, y_pred):
        i = classes.index(t)
        j = classes.index(p)
        cm[i, j] += 1

    return cm


def print_confusion_matrix_multiclass(cm, classes):

    print("\n" + "="*120)
    print("CONFUSION MATRIX KLASIFIKASI PENYAKIT DAUN TOMAT")
    print("="*120)

    # Header kolom
    print("{:25s}".format("Aktual \\ Prediksi"), end="")
    for cls in classes:
        print("{:15s}".format(cls[:12]), end="")
    print()

    # Baris tabel
    for i, cls in enumerate(classes):
        print("{:25s}".format(cls), end="")
        for j in range(len(classes)):
            print("{:15d}".format(cm[i, j]), end="")
        print()

    print("="*120)


def analyze_confusion_matrix_multiclass(cm, classes):

    print("\nANALISIS CONFUSION MATRIX PER KELAS")
    print("-"*120)

    for i, cls in enumerate(classes):

        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        print(f"\nKelas : {cls}")
        print(f"  True Positive  : {tp}")
        print(f"  False Negative : {fn}")
        print(f"  False Positive : {fp}")
        print(f"  Precision      : {precision:.4f}")
        print(f"  Recall         : {recall:.4f}")


# ================= CETAK CONFUSION MATRIX =================

cm = confusion_matrix_multiclass(y_true, y_pred, classes)

print_confusion_matrix_multiclass(cm, classes)
analyze_confusion_matrix_multiclass(cm, classes)

print("\nEvaluasi + confusion matrix multi-kelas selesai.")
