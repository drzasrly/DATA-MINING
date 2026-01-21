import os, cv2, pickle
import numpy as np
from feature_manual import extract_features
from config import IMG_SIZE

DATASET = "tomato_disease_ready"

def prepare(split):
    X, y = [], []
    classes = sorted(os.listdir(os.path.join(DATASET, split)))

    for idx, cls in enumerate(classes):
        for f in os.listdir(os.path.join(DATASET, split, cls)):
            img = cv2.imread(os.path.join(DATASET, split, cls, f))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(extract_features(img))
            y.append(idx)

    X = np.array(X)
    y = np.array(y)

    pickle.dump((X, y, classes), open(f"features_fast{split}.pkl", "wb"))
    print(f"Fitur {split} tersimpan ({len(X)} data)")

prepare("train")
prepare("validation")
prepare("test")
