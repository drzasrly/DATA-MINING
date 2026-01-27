import os
import cv2
import pickle
import numpy as np
import pandas as pd
from feature_manual import extract_features
from config import MODEL_PATH, K

# =========================
# PATH
# =========================
IMAGE_PATH = "test_images"
GT_PATH    = "multileaf_groundtruth.csv"

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open(MODEL_PATH, "rb"))

train_features = model["X"]
train_labels   = model["y"]
classes        = model["classes"]
mean           = model["mean"]
std            = model["std"]

print("\nModel berhasil dimuat")
print("Jumlah data latih :", len(train_labels))
print("Jumlah kelas     :", len(classes))

# =========================
# LOAD GROUND TRUTH
# =========================
if not os.path.exists(GT_PATH):
    print("\nERROR: File multileaf_groundtruth.csv tidak ditemukan!")
    print("Silakan buat / generate ground truth terlebih dahulu.")
    exit()

gt_data = pd.read_csv(GT_PATH)

# dictionary: (filename, leaf_id) -> true_label
gt_dict = {}
for _, row in gt_data.iterrows():
    key = (row["filename"], int(row["leaf_id"]))
    gt_dict[key] = row["true_label"]

print("Jumlah ground truth daun :", len(gt_dict))

# =========================
# VARIABLE EVALUASI
# =========================
y_true = []
y_pred = []
total_leaf = 0
total_labeled = 0

print("\n=== EVALUASI MULTILEAF PER DAUN (WATERSHED BASED) ===\n")

# =========================
# LOOP IMAGE
# =========================
for file in sorted(os.listdir(IMAGE_PATH)):

    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_PATH, file)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image = cv2.resize(image, (512, 512))
    orig = image.copy()

    # =========================
    # SEGMENTASI WATERSHED
    # =========================
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

    # =========================
    # LOOP SETIAP DAUN
    # =========================
    leaf_id = 0

    for m in np.unique(markers_ws):

        if m <= 1:
            continue

        mask_leaf = np.uint8(markers_ws == m) * 255
        x, y, w, h = cv2.boundingRect(mask_leaf)

        if w < 30 or h < 30:
            continue

        leaf_id += 1
        total_leaf += 1

        leaf_roi = cv2.resize(orig[y:y+h, x:x+w], (256, 256))

        # =========================
        # FEATURE & NORMALISASI
        # =========================
        feat = np.array(extract_features(leaf_roi))
        feat = (feat - mean) / (std + 1e-8)

        # =========================
        # KNN MANUAL
        # =========================
        distances = [np.sqrt(np.sum((train_features[i] - feat) ** 2))
                     for i in range(len(train_features))]

        k_indices = np.argsort(distances)[:K]
        k_labels = [classes[train_labels[i]] for i in k_indices]

        pred_label = max(set(k_labels), key=k_labels.count)

        # =========================
        # AMBIL GROUND TRUTH
        # =========================
        key = (file, leaf_id)

        if key in gt_dict:
            true_label = gt_dict[key]

            y_true.append(true_label)
            y_pred.append(pred_label)
            total_labeled += 1

            print(f"{file:20s} | Daun-{leaf_id:03d} | True: {true_label:20s} | Pred: {pred_label:20s}")

        else:
            print(f"{file:20s} | Daun-{leaf_id:03d} | Pred: {pred_label:20s} | (TIDAK ADA GROUND TRUTH)")

# =========================
# HITUNG ACCURACY MANUAL
# =========================
def accuracy_manual(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true) if len(y_true) > 0 else 0

accuracy = accuracy_manual(y_true, y_pred) * 100

# =========================
# OUTPUT HASIL
# =========================
print("\n" + "="*70)
print("HASIL EVALUASI MULTILEAF PER DAUN")
print("="*70)

print(f"Total daun terdeteksi : {total_leaf}")
print(f"Total daun berlabel  : {total_labeled}")
print(f"Prediksi benar       : {sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])}")
print(f"Accuracy             : {accuracy:.2f}%")

print("="*70)
print("Evaluasi selesai.")
