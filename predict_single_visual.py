import cv2, pickle, numpy as np
from feature_manual import extract_features
from knn_manual import knn_predict
from config import MODEL_PATH, K
from utils_visual import make_canvas

model = pickle.load(open(MODEL_PATH, "rb"))

img = cv2.imread("test_images/late_blight.jpg")
img = cv2.resize(img, (256,256))
orig = img.copy()

# HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Mask
lower = np.array([20,30,30])
upper = np.array([90,255,255])
mask = cv2.inRange(hsv, lower, upper)

mask_clean = cv2.medianBlur(mask, 5)

# Feature & KNN
feat = np.array(extract_features(orig))
feat = (feat - model['mean']) / model['std']
pred = knn_predict(model['X'], model['y'], feat, K)
label = model['classes'][pred]

cv2.putText(orig, label, (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

from utils_visual import make_canvas, add_title

canvas = make_canvas([
    [
        (orig, "Original Image + Prediction"),
        (h, "Hue (HSV)"),
        (s, "Saturation (HSV)")
    ],
    [
        (v, "Value (HSV)"),
        (mask, "HSV Mask"),
        (mask_clean, "Mask + Median Filter")
    ]
])



cv2.imshow("Predict Single - All Stages", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
