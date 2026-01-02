import cv2, pickle, numpy as np
from feature_manual import extract_features
from knn_manual import knn_predict
from config import MODEL_PATH, K
from utils_visual import make_canvas

model = pickle.load(open(MODEL_PATH, "rb"))

image = cv2.imread("test_images/multi_daun1.jpg")
image = cv2.resize(image, (256,256))
orig = image.copy()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

lower = np.array([20, 30, 30])
upper = np.array([90, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

mask = cv2.medianBlur(mask, 5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
dist_norm = np.uint8(dist_norm)

_, fg = cv2.threshold(dist, 0.3 * dist.max(), 255, cv2.THRESH_BINARY)
fg = np.uint8(fg)

unknown = cv2.subtract(mask, fg)
_, markers = cv2.connectedComponents(fg)
markers = markers + 1
markers[unknown == 255] = 0

markers_ws = cv2.watershed(image, markers)

leaf_count = 0
final = orig.copy()

for m in np.unique(markers_ws):
    if m <= 1:
        continue

    mask_leaf = np.uint8(markers_ws == m) * 255
    x, y, w, h_box = cv2.boundingRect(mask_leaf)

    if w < 30 or h_box < 30:
        continue

    leaf = orig[y:y+h_box, x:x+w]

    feat = np.array(extract_features(leaf))
    feat = (feat - model['mean']) / model['std']
    pred = knn_predict(model['X'], model['y'], feat, K)
    label = model['classes'][pred]

    leaf_count += 1
    cv2.rectangle(final, (x,y), (x+w,y+h_box), (0,255,0), 2)
    cv2.putText(final, label, (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

print("Total daun terdeteksi:", leaf_count)

canvas = make_canvas([
    [
        (orig, "Original Image"),
        (h, "Hue (HSV)"),
        (s, "Saturation (HSV)")
    ],
    [
        (v, "Value (HSV)"),
        (mask, "Mask HSV (Cleaned)"),
        (dist_norm, "Distance Transform")
    ],
    [
        (fg, "Foreground"),
        (markers_ws.astype(np.uint8)*10, "Watershed Markers"),
        (final, "Final Detection + KNN")
    ]
])

cv2.imshow("Predict Multi Leaf - All Stages", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
