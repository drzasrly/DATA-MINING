import os, cv2, pickle
import numpy as np
from feature_manual import extract_features
from config import TRAIN_DIR, IMG_SIZE, MODEL_PATH


classes = sorted(os.listdir(TRAIN_DIR))
X, y = [], []


for idx, cls in enumerate(classes):
    for f in os.listdir(f"{TRAIN_DIR}/{cls}"):
        img = cv2.imread(f"{TRAIN_DIR}/{cls}/{f}")
        if img is None: continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(extract_features(img))
        y.append(idx)

X = np.array(X)
y = np.array(y)


mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-8
Xn = (X - mean) / std


model = {"X": Xn, "y": y, "mean": mean, "std": std, "classes": classes}


os.makedirs("models", exist_ok=True)
pickle.dump(model, open(MODEL_PATH, "wb"))
print("Model manual tersimpan")