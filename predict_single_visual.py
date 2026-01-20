import cv2, pickle, numpy as np
import matplotlib.pyplot as plt
from feature_manual import extract_features, compute_glcm
from knn_manual import knn_predict
from config import MODEL_PATH, K
from utils_visual import make_canvas, add_title

model = pickle.load(open(MODEL_PATH, "rb"))

img = cv2.imread("test_images/late_blight.jpg")
img = cv2.resize(img, (256,256))
orig = img.copy()

# =========================
# HSV Processing
# =========================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

lower = np.array([20,30,30])
upper = np.array([90,255,255])
mask = cv2.inRange(hsv, lower, upper)
mask_clean = cv2.medianBlur(mask, 5)

# =========================
# Feature & KNN
# =========================
feat = np.array(extract_features(orig))
feat = (feat - model['mean']) / model['std']
pred = knn_predict(model['X'], model['y'], feat, K)
label = model['classes'][pred]

cv2.putText(orig, f"Prediction: {label}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

# =========================
# GLCM Visualization
# =========================
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
glcm = compute_glcm(gray, distance=1, angle=0)

glcm_norm = (glcm / glcm.max() * 255).astype(np.uint8)

# =========================
# Canvas Layout (rapi)
# =========================
canvas = make_canvas([
    [
        (orig, "Original + Prediction"),
        (mask_clean, "Leaf Mask"),
        (glcm_norm, "GLCM Matrix")
    ],
    [
        (h, "Hue"),
        (s, "Saturation"),
        (v, "Value")
    ]
])

cv2.imshow("Predict Single - Visual Pipeline", canvas)

# =========================
# HSV Histogram
# =========================
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.hist(h.flatten(), bins=180)
plt.title("Hue Histogram")

plt.subplot(1,3,2)
plt.hist(s.flatten(), bins=256)
plt.title("Saturation Histogram")

plt.subplot(1,3,3)
plt.hist(v.flatten(), bins=256)
plt.title("Value Histogram")

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
