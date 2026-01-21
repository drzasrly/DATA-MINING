import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def proses_lengkap_tomat(image_path):
    # --- 1. LOAD GAMBAR ---
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' tidak ditemukan!")
        return

    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256)) 
    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- 2. PREPROCESSING (Masking Hijau & Pembersihan) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Range warna yang sedikit lebih luas untuk menangkap variasi warna penyakit
    lower_green = np.array([15, 30, 30]) 
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # PERBAIKAN: Gunakan Median Blur untuk menghilangkan noise bintik (salt-and-pepper)
    mask = cv2.medianBlur(mask, 5)

    # PERBAIKAN: Gunakan kombinasi OPEN dan CLOSE dengan kernel Ellipse
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # Opening: menghilangkan noise kecil di luar objek
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # Closing: menutup lubang-lubang kecil di dalam objek daun
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- 3. WATERSHED SEGMENTATION (Optimasi Pemisahan) ---
    # Area yang pasti background (Sure BG)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    # Distance Transform: Semakin jauh dari tepi, nilai semakin tinggi (titik pusat daun)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # PERBAIKAN: Threshold 0.3 * max biasanya lebih efektif memisahkan daun yang rimbun
    # Dibandingkan 0.4, nilai 0.3 lebih sensitif dalam mendeteksi pusat daun yang kecil
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Area perbatasan yang akan dicari oleh Watershed (Unknown)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Labeling Marker
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 
    markers[unknown == 255] = 0 

    # Algoritma Watershed
    markers_ws = cv2.watershed(img, markers.copy())
    
    # --- 4. ROI EXTRACTION & VISUALISASI ---
    img_hasil_kotak = original_rgb.copy()
    leaf_count = 0
    
    os.makedirs("hasil_roi", exist_ok=True)

    for label in np.unique(markers_ws):
        if label <= 1: 
            continue
        
        mask_daun = np.uint8(markers_ws == label)
        contours, _ = cv2.findContours(mask_daun, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            
            # Filter objek yang terlalu kecil (noise yang tersisa)
            if w > 25 and h > 25:
                leaf_count += 1
                roi = original_rgb[y:y+h, x:x+w]
                
                # Simpan ROI
                cv2.imwrite(f"hasil_roi/daun_{leaf_count}.jpg", cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                
                # Gambar kotak
                cv2.rectangle(img_hasil_kotak, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_hasil_kotak, f"ID:{leaf_count}", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # --- 5. TAMPILKAN HASIL 5 PANEL ---
    print(f"Berhasil mendeteksi {leaf_count} daun.")
    
    plt.figure(figsize=(18, 6))
    tahapan = [
        (original_rgb, "1. Original"),
        (mask, "2. Masking (Cleaned)"),
        (dist_transform, "3. Distance Map"),
        (markers_ws, "4. Watershed Markers"),
        (img_hasil_kotak, "5. ROI Detection")
    ]

    for i, (gambar, judul) in enumerate(tahapan):
        plt.subplot(1, 5, i+1)
        if judul == "3. Distance Map" or judul == "4. Watershed Markers":
            plt.imshow(gambar, cmap='jet')
        else:
            plt.imshow(gambar)
        plt.title(judul)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Masukkan path gambar Anda
proses_lengkap_tomat("test_images/multi_daun1.jpg")