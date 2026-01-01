import cv2, pickle, numpy as np
from preprocessing_visual import preprocess_visual
from feature_manual import extract_features
from knn_manual import knn_predict
from config import MODEL_PATH, K

# =============================
# LOAD MODEL
# =============================
model = pickle.load(open(MODEL_PATH, "rb"))

# =============================
# LOAD & RESIZE IMAGE
# =============================
image = cv2.imread("test_images/multi_daun1.jpg")
image = cv2.resize(image, (256,256))
orig = image.copy()

# =============================
# HSV & VISUALISASI CHANNEL
# =============================
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

cv2.imshow("Original", image)
cv2.imshow("Hue", h)
cv2.imshow("Saturation", s)
cv2.imshow("Value", v)

# =============================
# SEGMENTASI HSV (LEBIH LONGGAR)
# =============================
lower = np.array([20, 30, 30])
upper = np.array([90, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
cv2.imshow("Mask HSV", mask)

# =============================
# NOISE REMOVAL (PENTING)
# =============================
mask = cv2.medianBlur(mask, 5)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Mask Cleaned", mask)

# =============================
# DISTANCE TRANSFORM
# =============================
dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
cv2.imshow("Distance Transform", dist)

# =============================
# FOREGROUND (LEBIH AGRESIF)
# =============================
_, fg = cv2.threshold(dist, 0.3, 1.0, cv2.THRESH_BINARY)
fg = np.uint8(fg * 255)
cv2.imshow("Foreground", fg)

# =============================
# BACKGROUND
# =============================
unknown = cv2.subtract(mask, fg)

# =============================
# MARKERS
# =============================
_, markers = cv2.connectedComponents(fg)
markers = markers + 1
markers[unknown == 255] = 0

# =============================
# WATERSHED
# =============================
markers = cv2.watershed(image, markers)

# =============================
# LOOP SETIAP DAUN
# =============================
leaf_count = 0
for m in np.unique(markers):
    if m <= 1:
        continue

    mask_leaf = np.uint8(markers == m) * 255
    x, y, w, h = cv2.boundingRect(mask_leaf)

    if w < 30 or h < 30:
        continue

    leaf = orig[y:y+h, x:x+w]

    # ===== KNN =====
    feat = np.array(extract_features(leaf))
    feat = (feat - model['mean']) / model['std']
    pred = knn_predict(model['X'], model['y'], feat, K)
    label = model['classes'][pred]

    leaf_count += 1
    cv2.rectangle(orig, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(orig, label, (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

print("Total daun terdeteksi:", leaf_count)

cv2.imshow("Multi Leaf Final Result", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
