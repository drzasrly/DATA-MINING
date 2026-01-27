import cv2, pickle, numpy as np
from feature_manual import extract_features
from config import MODEL_PATH, K

# ================= 1. LOAD MODEL & SETTINGS =================
model = pickle.load(open(MODEL_PATH, "rb"))

# Warna untuk estetika (BGR)
GREEN_LIME = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# ================= 2. PREPROCESS IMAGE =================
image = cv2.imread("test_images/multi_daun1.jpg")
if image is None:
    exit("Error: Gambar tidak ditemukan!")

# Gunakan resolusi yang cukup besar agar detail tidak pecah
image = cv2.resize(image, (800, 600)) 
orig = image.copy()
final = orig.copy()

# Segmentasi Watershed (Logika tetap sama)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([90, 255, 255]))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
_, fg = cv2.threshold(dist, 0.3 * dist.max(), 255, cv2.THRESH_BINARY)
unknown = cv2.subtract(mask, np.uint8(fg))
_, markers = cv2.connectedComponents(np.uint8(fg))
markers = markers + 1
markers[unknown == 255] = 0
markers_ws = cv2.watershed(image, markers)

# ================= 3. PREDICT & STYLED DRAWING =================
leaf_count = 0

for m in np.unique(markers_ws):
    if m <= 1: continue

    mask_leaf = np.uint8(markers_ws == m) * 255
    x, y, w, h_box = cv2.boundingRect(mask_leaf)
    if w < 40 or h_box < 40: continue

    # Predict Logic
    leaf_resized = cv2.resize(orig[y:y+h_box, x:x+w], (256, 256))
    feat = (np.array(extract_features(leaf_resized)) - model['mean']) / (model['std'] + 1e-8)
    distances = [np.sqrt(np.sum((model['X'][i] - feat) ** 2)) for i in range(len(model['X']))]
    k_labels = [model['classes'][model['y'][i]] for i in np.argsort(distances)[:K]]
    pred_label = max(set(k_labels), key=k_labels.count)
    confidence = (k_labels.count(pred_label) / K) * 100
    leaf_count += 1

    # --- STYLE BARU: DRAWING BOX & TEXT BAR ---
    # 1. Gambar Bounding Box yang lebih halus
    cv2.rectangle(final, (x, y), (x + w, y + h_box), GREEN_LIME, 2)

    # 2. Siapkan Teks Label
    label_txt = f"ID:{leaf_count} {pred_label.replace('_', ' ').title()} ({confidence:.0f}%)"
    font = cv2.FONT_HERSHEY_DUPLEX # Font lebih tajam dari Simplex
    font_scale = 0.5
    thickness = 1
    
    # 3. Buat Background Bar untuk Teks (agar terbaca jelas)
    (text_w, text_h), baseline = cv2.getTextSize(label_txt, font, font_scale, thickness)
    cv2.rectangle(final, (x, y - text_h - 10), (x + text_w + 10, y), GREEN_LIME, -1) # Box Hijau Solid
    
    # 4. Tulis Teks di atas Box Hijau (Warna Hitam agar kontras)
    cv2.putText(final, label_txt, (x + 5, y - 7), font, font_scale, BLACK, thickness, cv2.LINE_AA)

# ================= 4. DISPLAY =================
print(f"\n[INFO] Berhasil mengidentifikasi {leaf_count} daun.")

cv2.imshow("Hasil Klasifikasi Multi-Leaf - Standard Jurnal", final)
cv2.imwrite("hasil_final_jurnal.png", final) # Hasil kualitas tinggi tersimpan otomatis
cv2.waitKey(0)
cv2.destroyAllWindows()