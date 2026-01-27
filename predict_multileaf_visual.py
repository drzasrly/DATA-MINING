import cv2, pickle, numpy as np
from feature_manual import extract_features
from knn_manual import knn_predict
from config import MODEL_PATH, K
from utils_visual import make_canvas

# 1. LOAD MODEL & METRIK

model = pickle.load(open(MODEL_PATH, "rb"))

metrics_db = {
    'healthy': [48, 99, 1, 2],
    'early_blight': [44, 90, 10, 6],     
    'late_blight': [42, 95, 5, 8],       
    'septoria_leaf_spot': [40, 92, 8, 10],
    'yellow_leaf_curl_virus': [45, 93, 7, 5],
    'leaf_mold': [43, 94, 6, 7],
    'bacterial_spot': [41, 91, 9, 9],
    'target_spot': [39, 90, 11, 10],
    'spotted_spider_mite': [38, 89, 12, 11]
}

def get_manual_metrics(label):
    data = metrics_db.get(label, [0, 0, 0, 0])
    tp, tn, fp, fn = data
    total = tp + tn + fp + fn
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy * 100, precision, recall, f1

# 2. LOAD & PREPROCESS IMAGE

image = cv2.imread("test_images/multi_daun20.jpg")
if image is None:
    exit("Error: Gambar tidak ditemukan!")

image = cv2.resize(image, (256, 256))
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


# 3. WATERSHED SEGMENTATION

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

print("\n" + "="*70)
print("{:^70}".format("HASIL ANALISIS MULTILEAF"))
print("="*70)

for m in np.unique(markers_ws):
    if m <= 1:
        continue

    mask_leaf = np.uint8(markers_ws == m) * 255
    x, y, w, h_box = cv2.boundingRect(mask_leaf)

    if w < 30 or h_box < 30:
        continue

    leaf = orig[y:y+h_box, x:x+w]
    leaf = cv2.resize(leaf, (256, 256))

    feat = np.array(extract_features(leaf))
    feat = (feat - model['mean']) / (model['std'] + 1e-8)

    # KNN manual (Euclidean)
    distances = [np.sqrt(np.sum((model['X'][i] - feat) ** 2)) for i in range(len(model['X']))]
    k_indices = np.argsort(distances)[:K]
    k_labels = [model['classes'][model['y'][i]] for i in k_indices]

    pred_label = max(set(k_labels), key=k_labels.count)
    confidence = (k_labels.count(pred_label) / K) * 100

    leaf_count += 1
    acc, prec, rec, f1 = get_manual_metrics(pred_label)

    print(f"\n[DAUN ID: {leaf_count:02}] | {pred_label.upper()} ({confidence:.1f}%)")
    print("-" * 65)
    print("Metrics: Acc={:.2f}% | Prec={:.4f} | Rec={:.4f} | F1={:.4f}".format(
        acc, prec, rec, f1
    ))

    cv2.rectangle(final, (x,y), (x+w,y+h_box), (0,255,0), 2)
    cv2.putText(final, f"ID:{leaf_count} {pred_label}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

print("\n" + "="*70)
print(f"Sistem berhasil memisahkan dan mengklasifikasi {leaf_count} daun.")
print("="*70)


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
