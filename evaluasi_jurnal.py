import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_PATH, K

# ================= 1. LOAD DATA & MODEL =================

def load_all():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def load_features(feature_file):
    with open(feature_file, "rb") as f:
        X, y, _ = pickle.load(f)
    return X, y

# ================= 2. CORE LOGIC =================

def knn_fast(X_train, y_train, x_test, k):
    distances = np.linalg.norm(X_train - x_test, axis=1)
    k_idx = np.argsort(distances)[:k]
    k_labels = y_train[k_idx]
    return np.bincount(k_labels).argmax()

def confusion_matrix_multiclass(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# ================= 3. HITUNG TP, TN, FP, FN =================

def print_detailed_metrics(cm, classes):
    """
    Menghitung dan mencetak TP, TN, FP, FN untuk setiap kelas
    """
    total_samples = np.sum(cm)
    
    print("\n" + "="*120)
    print(f"{'KELAS DAUN':<25} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} {'PRECISION':>12} {'RECALL':>10}")
    print("-" * 120)

    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total_samples - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"{cls[:25]:<25} {tp:6d} {tn:6d} {fp:6d} {fn:6d} {precision:12.4f} {recall:10.4f}")
    
    print("="*120)
    print(f"INFO: TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative")

# ================= 4. VISUALISASI HEATMAP =================

def plot_journal_cm(cm, classes, name):
    plt.figure(figsize=(12, 10))
    
    # Transpose agar Sumbu X = Aktual (Atas), Sumbu Y = Prediksi (Samping)
    cm_plot = cm.T 

    ax = sns.heatmap(cm_plot, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=classes, yticklabels=classes,
                    square=True, cbar_kws={'label': 'Jumlah Sampel'})
    
    plt.xlabel('ACTUALLY (DATA AKTUAL)', fontsize=12, fontweight='bold', labelpad=15)
    plt.ylabel('PREDICTED (HASIL PREDIKSI)', fontsize=12, fontweight='bold', labelpad=15)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0)
    
    plt.title(f'Confusion Matrix Standard Jurnal - {name}\n', fontsize=14, pad=35, fontweight='bold')
    
    output_name = f"cm_{name.lower()}.png"
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()
    print(f"[SUCCESS] Gambar CM disimpan: {output_name}")

# ================= 5. MAIN EXECUTION =================

def run_evaluation(feature_file, name, model):
    print(f"\n>>> MENGANALISIS DATASET: {name}")
    
    X_raw, y_true = load_features(feature_file)
    classes = model["classes"]
    
    # Normalisasi menggunakan parameter model
    X_norm = (X_raw - model["mean"]) / (model["std"] + 1e-8)

    y_pred = [knn_fast(model["X"], model["y"], x, K) for x in X_norm]

    # Hitung CM
    cm = confusion_matrix_multiclass(y_true, y_pred, len(classes))

    # Tampilkan Tabel TP, TN, FP, FN
    print_detailed_metrics(cm, classes)
    
    # Simpan Gambar
    plot_journal_cm(cm, classes, name)

if __name__ == "__main__":
    try:
        my_model = load_all()
        
        # Daftar file yang akan dievaluasi
        files = [
            ("features_fasttrain.pkl", "TRAIN"),
            ("features_fastvalidation.pkl", "VALIDATION"),
            ("features_fasttest.pkl", "TEST")
        ]
        
        for file, label in files:
            run_evaluation(file, label, my_model)
            
    except Exception as e:
        print(f"\n[ERROR] Terjadi kesalahan: {e}")
        print("Pastikan sudah install library: pip install seaborn matplotlib numpy")