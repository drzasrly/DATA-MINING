import cv2, pickle, os, numpy as np
from feature_manual import extract_features
from knn_manual import knn_predict
from config import MODEL_PATH, K, TEST_DIR

model = pickle.load(open(MODEL_PATH, "rb"))

correct = 0
total = 0

for cls in os.listdir(TEST_DIR):
    cls_path = os.path.join(TEST_DIR, cls)
    for file in os.listdir(cls_path):
        img_path = os.path.join(cls_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (256,256))
        feat = np.array(extract_features(img))
        feat = (feat - model['mean']) / model['std']

        pred = knn_predict(model['X'], model['y'], feat, K)
        pred_label = model['classes'][pred]

        if pred_label == cls:
            correct += 1
        total += 1

accuracy = correct / total * 100
print(f"Akurasi sistem: {accuracy:.2f}%")
