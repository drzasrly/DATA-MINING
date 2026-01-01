import cv2, pickle
import numpy as np
from preprocessing_visual import preprocess_visual
from feature_manual import extract_features
from knn_manual import knn_predict
from config import MODEL_PATH, K


model = pickle.load(open(MODEL_PATH, "rb"))
img = cv2.imread("test_images/late_blight.jpg")
seg, _ = preprocess_visual(img)


feat = np.array(extract_features(seg))
feat = (feat - model['mean']) / model['std']


pred = knn_predict(model['X'], model['y'], feat, K)
print("Hasil klasifikasi:", model['classes'][pred])


cv2.waitKey(0)
cv2.destroyAllWindows()