import cv2, pickle, os, numpy as np
from feature_manual import extract_features
from knn_manual import knn_predict
from config import MODEL_PATH, K

model = pickle.load(open(MODEL_PATH, "rb"))

IMG_SIZE = 256

def evaluate_dataset(dataset_dir):
    correct = 0
    total = 0

    for cls in os.listdir(dataset_dir):
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            feat = np.array(extract_features(img))
            feat = (feat - model['mean']) / model['std']

            pred = knn_predict(model['X'], model['y'], feat, K)
            pred_label = model['classes'][pred]

            if pred_label == cls:
                correct += 1
            total += 1

    acc = (correct / total) * 100 if total > 0 else 0
    return acc, total


train_acc, train_total = evaluate_dataset("tomato_disease_ready/train")
val_acc, val_total     = evaluate_dataset("tomato_disease_ready/validation")
test_acc, test_total   = evaluate_dataset("tomato_disease_ready/test")

print("=== HASIL EVALUASI SISTEM ===")
print(f"Train Accuracy      : {train_acc:.2f}% ({train_total} data)")
print(f"Validation Accuracy : {val_acc:.2f}% ({val_total} data)")
print(f"Test Accuracy       : {test_acc:.2f}% ({test_total} data)")
