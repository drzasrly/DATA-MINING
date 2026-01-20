import os, cv2, pickle
import numpy as np
import pandas as pd
from feature_manual import extract_features
from config import TRAIN_DIR, IMG_SIZE, MODEL_PATH

classes = sorted(os.listdir(TRAIN_DIR))
X, y, filenames, class_names = [], [], [], []

for idx, cls in enumerate(classes):
    cls_path = os.path.join(TRAIN_DIR, cls)
    for f in os.listdir(cls_path):
        img_path = os.path.join(cls_path, f)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        feat = extract_features(img)

        X.append(feat)
        y.append(idx)
        filenames.append(f)
        class_names.append(cls)

X = np.array(X)
y = np.array(y)

# =========================
# Normalisasi
# =========================
mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-8
Xn = (X - mean) / std

model = {
    "X": Xn,
    "y": y,
    "mean": mean,
    "std": std,
    "classes": classes
}

os.makedirs("models", exist_ok=True)
pickle.dump(model, open(MODEL_PATH, "wb"))
print("Model manual tersimpan")

# =========================
# Simpan ke CSV
# =========================
feature_names = [f"f{i+1}" for i in range(X.shape[1])]

df = pd.DataFrame(X, columns=feature_names)
df["label"] = y
df["class_name"] = class_names
df["filename"] = filenames

os.makedirs("exports", exist_ok=True)
csv_path = "exports/features_dataset.csv"
df.to_csv(csv_path, index=False)

print(f"CSV fitur tersimpan di: {csv_path}")
