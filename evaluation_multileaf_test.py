import os
import cv2
import pickle
import numpy as np
from collections import Counter
from feature_manual import extract_features
from config import MODEL_PATH, K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

y_true = []
y_pred = []
conf_list = []
total_leaf = 0

print("\n=== EVALUASI MULTILEAF PER DAUN (WATERSHED BASED) ===\n")

for file in sorted(os.listdir(IMAGE_PATH)):

    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_PATH, file)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image = cv2.resize(image, (512, 512))
    orig = image.copy()

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

print("\n" + "="*75)
print("HASIL EVALUASI MULTILEAF PER DAUN (WATERSHED BASED)")
print("="*75)

accuracy  = accuracy_score(y_true, y_pred) * 100
precision = precision_score(y_true, y_pred, average="macro")
recall    = recall_score(y_true, y_pred, average="macro")
f1        = f1_score(y_true, y_pred, average="macro")

print(f"Total Daun Terdeteksi : {total_leaf}")
print(f"Accuracy             : {accuracy:.2f}%")
print(f"Precision            : {precision:.4f}")
print(f"Recall               : {recall:.4f}")
print(f"F1-Score             : {f1:.4f}")
print(f"Confidence Rata-rata : {np.mean(conf_list):.2f}%")

print("="*75)
print("Evaluasi multileaf selesai.")
