import os
import cv2
import numpy as np
import pandas as pd
import random
from feature_manual import extract_features

DATASET_PATH = "tomato_disease_ready/train"   
IMG_SIZE = 256
SAMPLES_PER_CLASS = 2            

rows = []
no = 1

for cls in sorted(os.listdir(DATASET_PATH)):
    class_path = os.path.join(DATASET_PATH, cls)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)[:SAMPLES_PER_CLASS]
    
    num_to_sample = min(len(images), SAMPLES_PER_CLASS)
    images = random.sample(images, num_to_sample)

    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        features = extract_features(img)

        row = {
            "No": no,
            "Hue Mean": round(features[0], 2),
            "Sat Mean": round(features[1], 2),
            "Val Mean": round(features[2], 2),
            "Contrast": round(features[6], 2),
            "Energy": round(features[8], 2),
            "Label": cls
        }

        rows.append(row)
        no += 1

df = pd.DataFrame(rows)

print("\n=== TABEL FITUR DATA LATIH (CONTOH) ===\n")
print(df.to_string(index=False))

df.to_csv("tabel_fitur_dataset.csv", index=False)
print("\n[INFO] Tabel disimpan sebagai tabel_fitur_dataset.csv")
