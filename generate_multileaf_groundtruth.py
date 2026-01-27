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
OUT_PATH   = "multileaf_groundtruth.csv"

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

rows = []

print("\n=== GENERATE GROUND TRUTH OTOMATIS (PREDIKSI AWAL) ===\n")

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
    # LOOP DAUN
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

        leaf_roi = cv2.resize(orig[y:y+h, x:x+w], (256, 256))

        # =========================
        # FEATURE & KNN
        # =========================
        feat = np.array(extract_features(leaf_roi))
        feat = (feat - mean) / (std + 1e-8)

        distances = [np.sqrt(np.sum((train_features[i] - feat) ** 2))
                     for i in range(len(train_features))]

        k_indices = np.argsort(distances)[:K]
        k_labels = [classes[train_labels[i]] for i in k_indices]

        pred_label = max(set(k_labels), key=k_labels.count)

        # =========================
        # SIMPAN KE CSV
        # =========================
        rows.append([file, leaf_id, pred_label])

        print(f"{file:20s} | Daun-{leaf_id:03d} | Pred awal: {pred_label}")

# =========================
# SIMPAN FILE
# =========================
df = pd.DataFrame(rows, columns=["filename", "leaf_id", "true_label"])
df.to_csv(OUT_PATH, index=False)

print("\n" + "="*70)
print("GROUND TRUTH OTOMATIS BERHASIL DIBUAT")
print("File :", OUT_PATH)
print("Jumlah daun :", len(rows))
print("="*70)
print("\nSilakan EDIT file CSV ini secara MANUAL untuk koreksi label.")
