import os
from collections import OrderedDict

BASE_PATH = "tomato_disease_ready"   # ganti sesuai folder Anda
SPLITS = ["train", "validation", "test"]

result = OrderedDict()

# Inisialisasi kelas dari folder train
classes = sorted(os.listdir(os.path.join(BASE_PATH, "train")))

for cls in classes:
    result[cls] = {"train": 0, "validation": 0, "test": 0, "total": 0}

# Hitung jumlah citra
for split in SPLITS:
    split_path = os.path.join(BASE_PATH, split)
    
    for cls in classes:
        class_path = os.path.join(split_path, cls)
        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])
            
            result[cls][split] = count
            result[cls]["total"] += count

# Cetak tabel seperti jurnal
print("\n=== DISTRIBUSI DATASET PER KELAS ===\n")
print(f"{'Kelas':25s} {'Total':>8s} {'Train':>8s} {'validation':>8s} {'Test':>8s}")
print("-"*65)

total_all = {"train":0, "validation":0, "test":0, "total":0}

for cls, data in result.items():
    print(f"{cls:25s} {data['total']:8d} {data['train']:8d} {data['validation']:8d} {data['test']:8d}")
    
    for k in total_all:
        total_all[k] += data[k]

print("-"*65)
print(f"{'TOTAL':25s} {total_all['total']:8d} {total_all['train']:8d} {total_all['validation']:8d} {total_all['test']:8d}")
